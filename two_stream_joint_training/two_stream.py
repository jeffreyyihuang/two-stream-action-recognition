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

from util import *
from network import *
from dataloader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch Sub-JHMDB rgb frame training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
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
    
    train_loader = data_loader.train()
    test_loader = data_loader.validate()
    #Model 
    model = Model(
                        # Data Loader
                        train_loader=train_loader,
                        test_loader=test_loader,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,)
    #Training
    model.run()

class Model():
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

    def build_model(self):
        
        print ('==> Build Spatial, Motion CNN')
        self.spatial_cnn = resnet101(pretrained= True, channel=3).cuda()
        self.motion_cnn = resnet101(pretrained= True, channel=20).cuda()
        #print self.model
        #Loss function and optimizer
        print ('==> Setup loss function and optimizer')
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.S_optimizer = torch.optim.SGD(self.spatial_cnn.parameters(), lr=5e-6, momentum=0.9)
        self.M_optimizer = torch.optim.SGD(self.motion_cnn.parameters(), lr=5e-3, momentum=0.9)
    '''
    def resume_and_evaluate(self):
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
    '''    
    def run(self):
        self.build_model()
        #self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1 = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            # save model
            if is_best:
                self.best_prec1 = prec1
            
            save_checkpoint({
                'epoch': self.epoch,
                'rgb_state_dict': self.spatial_cnn.state_dict(),
                'motion_state_dict': self.motion_cnn.state_dict(),
                'best_prec1': self.best_prec1,
                'rgb_optimizer' : self.S_optimizer.state_dict(),
                'opf_optimizer' : self.M_optimizer.state_dict()
            },is_best)

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        S_losses = AverageMeter()
        M_losses = AverageMeter()
        rgb_top1 = AverageMeter()
        rgb_top5 = AverageMeter()
        opf_top1 = AverageMeter()
        opf_top5 = AverageMeter()
        #switch to train mode
        self.spatial_cnn.train() 
        self.motion_cnn.train()   
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (rgb,opf,label) in enumerate(progress):

            # measure data loading time
            data_time.update(time.time() - end)
             
            label = label.cuda(async=True)
            rgb_var = Variable(rgb).cuda()
            opf_var = Variable(opf).cuda()
            target_var = Variable(label).cuda()

            # compute output
            S_output = self.spatial_cnn(rgb_var)
            M_output = self.motion_cnn(opf_var)   
            S_loss = self.criterion(S_output, target_var)
            M_loss = self.criterion(M_output, target_var)

            # measure accuracy and record loss
            S_prec1, S_prec5 = accuracy(S_output.data, label, topk=(1, 5))
            M_prec1, M_prec5 = accuracy(M_output.data, label, topk=(1, 5))

            S_losses.update(S_loss.data[0], rgb.size(0))
            rgb_top1.update(S_prec1[0], rgb.size(0))
            rgb_top5.update(S_prec5[0], rgb.size(0))
            M_losses.update(M_loss.data[0], rgb.size(0))
            opf_top1.update(M_prec1[0], rgb.size(0))
            opf_top5.update(M_prec5[0], rgb.size(0))

            # compute gradient and do SGD step
            self.S_optimizer.zero_grad()
            S_loss.backward()
            self.S_optimizer.step()

            self.M_optimizer.zero_grad()
            M_loss.backward()
            self.M_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'rgb loss':[round(S_losses.avg,5)],
                'rgb Prec@1':[round(rgb_top1.avg,4)],
                'rgb Prec@5':[round(rgb_top5.avg,4)],
                'opf loss':[round(M_losses.avg,5)],
                'opf Prec@1':[round(opf_top1.avg,4)],
                'opf Prec@5':[round(opf_top5.avg,4)]
                }
        record_info(info, 'record/training.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))

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
            M_losses.update(M_loss.data[0], rgb.size(0))

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
                    fusion_preds[videoName] = S_preds[j,:] + M_preds[j,:]
                else:
                    spatial_preds[videoName] += S_preds[j,:]
                    motion_preds[videoName] += M_preds[j,:]
                    fusion_preds[videoName] += (S_preds[j,:] + M_preds[j,:])
                    
        #Frame to video level accuracy
        rgb_top1, rgb_top5 = frame2_video_level_accuracy(spatial_preds)
        opf_top1, opf_top5 = frame2_video_level_accuracy(motion_preds)
        fuse_top1, fuse_top5 = frame2_video_level_accuracy(fusion_preds)


        info = {'Epoch':[self.epoch],
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
        return fuse_top1

    

class Data_Loader():
    def __init__(self, BATCH_SIZE, num_workers, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        #load data dictionary
        with open(dic_path+'/Video_training.pickle','rb') as f:
            self.dic_training=pickle.load(f)
        f.close()

        with open(dic_path+'/frame_count.pickle','rb') as f:
            self.dic_nb_frame=pickle.load(f)
        f.close()

        with open(dic_path+'/dic_test25_motion.pickle','rb') as f:
            self.dic_testing=pickle.load(f)
        f.close()
       
    def train(self):
        training_set = UCF_training_dataset(
            dic_video=self.dic_training,
            dic_nb_frame=self.dic_nb_frame, 
            rgb_root='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
            opf_root='/home/ubuntu/data/UCF101/tvl1_flow/', 
            transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            ]))
        print '==> Training data :',len(training_set)

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = UCF_testing_dataset(
            ucf_list=self.dic_testing,
            rgb_root='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
            opf_root='/home/ubuntu/data/UCF101/tvl1_flow/',
            transform = transforms.Compose([
            transforms.CenterCrop(224),
            #transforms.ToTensor(),
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