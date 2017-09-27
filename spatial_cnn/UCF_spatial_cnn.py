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

from util import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch Sub-JHMDB rgb frame training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
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
        if self.evaluate:
            prec1, val_loss = self.validate_1epoch()

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
            self.train_1epoch()
            print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
            prec1, val_loss = self.validate_1epoch()
            self.scheduler.step(prec1)
            
            is_best = prec1 > self.best_prec1
            if is_best:
                self.best_prec1 = prec1
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best)
            
    def train_1epoch(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)]}
        record_info(info, 'record/training.csv','train')

    def validate_1epoch(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            loss = self.criterion(output, label_var)

            # measure loss
            losses.update(loss.data[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('/',1)[0]
                if videoName not in dic_video_level_preds.keys():
                    dic_video_level_preds[videoName] = preds[j,:]
                else:
                    dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5 = frame2_video_level_accuracy(dic_video_level_preds)
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        record_info(info, 'record/testing.csv','test')
        return video_top1, losses.avg

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



def frame2_video_level_accuracy(dic_video_level_preds):
    with open('/home/ubuntu/cvlab/pytorch/ucf101_two_stream/dic_video_label.pickle','rb') as f:
        dic_video_label = pickle.load(f)
    f.close()
        
    correct = 0
    video_level_preds = np.zeros((len(dic_video_level_preds),101))
    video_level_labels = np.zeros(len(dic_video_level_preds))
    ii=0
    for line in sorted(dic_video_level_preds.keys()):
    
        n,g = line.split('_',1)
        if n == 'HandstandPushups':
            name = 'HandStandPushups_'+g
        else:
            name = line
        preds = dic_video_level_preds[line]
        label = int(dic_video_label[name])-1
            
        video_level_preds[ii,:] = preds
        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(preds) == (label):
            correct+=1

    #top1 top5
    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        
    top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                        
    top1 = float(top1.numpy())
    top5 = float(top5.numpy())
        
    #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
    return top1,top5

if __name__=='__main__':
    main()