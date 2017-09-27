import numpy as np
import pickle
from PIL import Image
import time
from tqdm import tqdm
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

from util import *
from network import *
from dataloader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch test network performance on UCF101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')

#parser.add_argument('--nb_per_stack', default=10, type=int, metavar='N',help='number of joint positions in 1 stack (default: 5)')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    #Prepare DataLoader
    data_loader = Data_Loader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        dic_path='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/dictionary/motion/'
                        )
    
    test_loader = data_loader.val()
    #Model 
    model = Model(
                        # Data Loader
                        test_loader=test_loader,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        spatial_weight='resnet101_weights/spatial_cnn_82.tar',
                        motion_weight='resnet101_weights/motion_cnn_78.tar'
                        )
    #Training
    model.run()

class Model():
    def __init__(self, nb_epochs, lr, batch_size, test_loader, spatial_weight, motion_weight):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.test_loader=test_loader
        self.spatial_weight=spatial_weight
        self.motion_weight=motion_weight

    def build_model(self):
        print ('==> Build Spatial, Motion CNN')
        self.spatial_cnn = resnet101(pretrained= True, channel=3).cuda()
        self.motion_cnn = resnet101(pretrained= True, channel=20).cuda()
        #Loss function and optimizer
        print ('==> Setup loss function and optimizer')
        self.criterion = nn.CrossEntropyLoss().cuda()

    def load_weight(self):
        s = torch.load(self.spatial_weight)
        m = torch.load(self.motion_weight)

        pretrain_dict = s['state_dict']
        model_dict=self.spatial_cnn.state_dict()
        weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
        weight_dict['conv1_custom.weight'] = pretrain_dict['conv1.weight']

        self.spatial_cnn.load_state_dict(weight_dict)
        print("==> load spatial cnn (epoch {}) (best_prec1 {})"
                  .format(s['epoch'], s['best_prec1']))

        self.motion_cnn.load_state_dict(m['state_dict'])
        print("==> load spatial cnn (epoch {}) (best_prec1 {})"
                  .format(m['epoch'], m['best_prec1']))
 
    def run(self):
        self.build_model()
        self.load_weight()
        cudnn.benchmark = True
        prec1 = self.validate_1epoch()
    
    def validate_1epoch(self):
        print('==> [validation stage]')

        batch_time = AverageMeter()
        S_losses = AverageMeter()
        M_losses = AverageMeter()

        # switch to evaluate mode
        self.spatial_cnn.eval() 
        self.motion_cnn.eval() 

        spatial_preds={}
        motion_preds={}
        fusion_preds={}

        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,rgb,opf,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            rgb_var = Variable(rgb, volatile=True).cuda(async=True)
            opf_var = Variable(opf, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            S_output = self.spatial_cnn(rgb_var)
            M_output = self.motion_cnn(opf_var)
             
            S_loss = self.criterion(S_output, label_var)
            M_loss = self.criterion(M_output, label_var)

            # measure accuracy and record loss
            S_losses.update(S_loss.data[0], rgb.size(0))
            M_losses.update(0, rgb.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            S_preds = S_output.data.cpu().numpy()
            M_preds = M_output.data.cpu().numpy()
        
            nb_data = S_preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j] # ApplyMakeup_g01_c01
                if videoName not in fusion_preds.keys():
                    spatial_preds[videoName] = S_preds[j,:]
                    motion_preds[videoName] = M_preds[j,:]
                    fusion_preds[videoName] = (S_preds[j,:] + M_preds[j,:])
                else:
                    spatial_preds[videoName] += S_preds[j,:]
                    motion_preds[videoName] += M_preds[j,:]
                    fusion_preds[videoName] += (S_preds[j,:] + M_preds[j,:])
                    
        #Frame to video level accuracy
        rgb_top1, rgb_top5 = frame2_video_level_accuracy(spatial_preds)
        opf_top1, opf_top5 = frame2_video_level_accuracy(motion_preds)
        fuse_top1, fuse_top5 = frame2_video_level_accuracy(fusion_preds)


        info = {
                'Batch Time':[round(batch_time.avg,3)],
                'rgb loss':[round(S_losses.avg,5)],
                'rgb Prec@1':[round(rgb_top1,4)],
                'rgb Prec@5':[round(rgb_top5,4)],
                'opf loss':[round(M_losses.avg,5)],
                'opf Prec@1':[round(opf_top1,4)],
                'opf Prec@5':[round(opf_top5,4)],
                'fuse Prec@1':[round(fuse_top1,4)],
                'fuse Prec@5':[round(fuse_top5,4)]
                }
        record_info(info, 'record/testing.csv','test')
        return  fuse_top1

    

class Data_Loader():
    def __init__(self, BATCH_SIZE, num_workers, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        #load data dictionary
        with open(dic_path+'/dic_test25_motion.pickle','rb') as f:
            self.dic_testing=pickle.load(f)
        f.close()
       
    def val(self):
        validation_set = UCF_testing_dataset(
            ucf_list=self.dic_testing,
            rgb_root='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
            opf_root='/home/ubuntu/data/UCF101/tvl1_flow/',
            transform = transforms.Compose([
            transforms.Scale(256),
            ]))
        print '==> Validation data :',len(validation_set)

        test_loader = DataLoader(
            dataset=validation_set, 
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
        for name in sorted(dic_video_level_preds.keys()):
            n,g = name.split('_',1)
            if n == 'HandstandPushups':
                name2 = 'HandStandPushups_'+g
            else:
                name2 =name

            preds = dic_video_level_preds[name]
            label = int(dic_video_label[name2])-1
                
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