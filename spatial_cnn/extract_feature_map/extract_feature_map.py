import numpy as np
import pickle
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset_util import *
from spatial_network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch Sub-JHMDB rgb frame training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-6, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--extract', default='', type=str, metavar='PATH', help='extract feature map before fc layer')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    #Prepare DataLoader
    data_loader = Data_Loader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=4,
                        data_path='/home/ubuntu/data/UCF101/spatial_no_sampled/',
                        dic_path='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/dictionary/spatial/', 
                        )
    
    train_loader = data_loader.train()
    test_loader = data_loader.validate()
    #Model 
    spatial_cnn = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
    )
    #Training
    spatial_cnn.run()

class Spatial_CNN():

    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.output_dir = '/home/ubuntu/data/UCF101/spatial_feature_map/'

    def run(self):
        self.model = resnet101(pretrained= True,nb_classes=101).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=0,verbose=True)

        cudnn.benchmark = True
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        
        self.extract_feature_map()

    def extract_feature_map(self):
        '''
        train_progress = tqdm(self.train_loader)
        for i, (keys,data,label) in enumerate(train_progress):

            input_var = Variable(data).cuda()
            output,feature_map = self.model(input_var)

            preds = feature_map.data.cpu().numpy()
            nb_data = preds.shape[0]
            
            for j in range(nb_data):
                name,idx = keys[j].split('[@]')
                folder = self.output_dir + name  
                dir = os.path.dirname(folder)
                try:
                    os.stat(dir)
                except:
                    os.makedirs(dir)
                with open(folder+idx+'.pickle','wb') as f:
                    pickle.dump(preds[j,:],f)
                f.close()
        '''
        test_progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(test_progress):

            input_var = Variable(data).cuda()
            output,feature_map = self.model(input_var)

            preds = feature_map.data.cpu().numpy()
            nb_data = preds.shape[0] 
            for j in range(nb_data):
                name,idx = keys[j].split('[@]')
                folder = self.output_dir + name  
                dir = os.path.dirname(folder)
                try:
                    os.stat(dir)
                except:
                    os.makedirs(dir)
                with open(folder+idx+'.pickle','wb') as f:
                    pickle.dump(preds[j,:],f)
                f.close()




       
            

class Data_Loader():
    def __init__(self, BATCH_SIZE, num_workers, data_path, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_path=data_path
        #load data dictionary
        with open(dic_path+'/dic_training.pickle','rb') as f:
            dic_training=pickle.load(f)
        f.close()

        with open(dic_path+'/dic_test25.pickle','rb') as f:
            dic_testing=pickle.load(f)
        f.close()

        self.training_set = UCF_training_set(dic=dic_training, root_dir=self.data_path, transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        self.validation_set = UCF_testing_set(ucf_list=dic_testing, root_dir=self.data_path ,transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(self.training_set)
        print '==> Validation data :',len(self.validation_set)

    def train(self):
        train_loader = DataLoader(
            dataset=self.training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        test_loader = DataLoader(
            dataset=self.validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return test_loader


if __name__=='__main__':
    main()