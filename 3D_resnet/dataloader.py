import numpy as np
import pickle
from PIL import Image
import time
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


class ResNet3D_training_set(Dataset):  
    def __init__(self, dic, nb_clips, root_dir, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.nb_clips= nb_clips
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        video = self.keys[idx]
        if video.split('/',1)[0] == 'HandStandPushups':
            #print video
            video = 'HandstandPushups/'+video.split('/',1)[1]

        clips_idx = randint(1,self.nb_clips[video])
        label = self.values[idx]
        label = int(label)-1
        data = torch.FloatTensor(3,16,112,112)
            
        for i in range(16):
            index = clips_idx + i
            if video.split('_')[0] == 'HandstandPushups':
                n,g = video.split('_',1)
                name = 'HandStandPushups_'+g
                path = self.root_dir + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
            else:
                path = self.root_dir + video.split('_')[0]+'/separated_images/v_'+video+'/v_'+video+'_'
         
            img = Image.open(path +str(index)+'.jpg')
            img = img.resize([112,112])
            data[:,i,:,:] = self.transform(img)
            img.close() 

        sample = (data,label)
        return sample

class ResNet3D_validation_set(Dataset):  
    def __init__(self, dic, root_dir, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        video,clip_idx = self.keys[idx].split('-')

        if video.split('/',1)[0] == 'HandStandPushups':
            video = 'HandstandPushups/'+video.split('/',1)[1]

        label = self.values[idx]
        label = int(label)-1
        data = torch.FloatTensor(3,16,112,112)
            
        for i in range(16):
            index = int(clip_idx) + i
            if video.split('_')[0] == 'HandstandPushups':
                n,g = video.split('_',1)
                name = 'HandStandPushups_'+g
                path = self.root_dir + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
            else:
                path = self.root_dir + video.split('_')[0]+'/separated_images/v_'+video+'/v_'+video+'_'
         
            img = Image.open(path +str(index)+'.jpg')
            img = img.resize([224,224])
            data[:,i,:,:] = self.transform(img)
            img.close() 

        sample = (video,data,label)
        return sample



class ResNet3D_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, data_path, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_path=data_path
        self.dic_nb_frame={}
        #load data dictionary
        with open(dic_path+'/train_video.pickle','rb') as f:
            self.train_video=pickle.load(f)
        f.close()

        with open(dic_path+'/frame_count.pickle','rb') as f:
            dic_frame=pickle.load(f)
        f.close()

        with open(dic_path+'/test_video.pickle','rb') as f:
            self.test_video=pickle.load(f)
        f.close()
        with open('/home/ubuntu/cvlab/pytorch/ucf101_two_stream/dic_video_label.pickle','rb') as f:
            self.dic_video_label = pickle.load(f)
        f.close()
        #Preprocessing
        for key in dic_frame:

            classname = key.split('_',1)[1].split('.',1)[0]
            self.dic_nb_frame[classname]= dic_frame[key]-16 # each segment with 16 frame
            #print self.dic_nb_frame

    def run(self):
        self.test_video_segment_labeling()
        self.train_video_labeling()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader
        

    def test_video_segment_labeling(self):
        self.dic_test_idx = {}
        for video in self.test_video:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                key = 'HandstandPushups_'+ g
            else:
                key=video
            nb_frame = int(self.dic_nb_frame[key])
            for idx in range(nb_frame):
                if idx % 16 ==0:
                    key = video + '-' + str(idx+1)
                    self.dic_test_idx[key] = self.dic_video_label[video]

    def train_video_labeling(self):
        self.dic_video_train={}
        for video in self.train_video:
            label = self.dic_video_label[video]
            self.dic_video_train[video] = label 
                            
    def train(self):
        training_set = ResNet3D_training_set(dic=self.dic_video_train, nb_clips=self.dic_nb_frame, root_dir=self.data_path, 
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]))
        print '==> Training data :',len(training_set),' videos'
        #print training_set[1]

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)

        return train_loader

    def val(self):
        validation_set = ResNet3D_validation_set(dic= self.dic_test_idx, root_dir=self.data_path ,
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]))
        print '==> Validation data :',len(validation_set),' clips'
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader


if __name__ == '__main__':
    data_loader = ResNet3D_DataLoader(BATCH_SIZE=1,num_workers=8,
                                        dic_path='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/resnet3d/dic/',
                                        data_path='/home/ubuntu/data/UCF101/spatial_no_sampled/'
                                        )
    train_loader,val_loader = data_loader.run()
    print type(train_loader),type(val_loader)