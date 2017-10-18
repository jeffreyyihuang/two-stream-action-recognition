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

 
class ResNet3D_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=112
        self.img_cols=112

    def stackopf(self):
        name = 'v_'+self.video
        u = self.root_dir+ 'u/' + name
        v = self.root_dir+ 'v/'+ name
        
        flow = torch.FloatTensor(2,self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)
        grc = GroupRandomCrop()
        grc.get_params()
        for j in range(1,self.in_channel+1):
            idx = str((i-1)+j)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))
            if self.mode == 'train':
                H = self.transform(grc.crop(imgH))
                V = self.transform(grc.crop(imgV))
            elif self.mode == 'val':
                H = self.transform(imgH)
                V = self.transform(imgV)
            else:
                raise ValueError('There are only train and val mode')
            
            flow[0,j-1,:,:] = H
            flow[1,j-1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = randint(1,int(nb_clips))
        elif self.mode == 'val':
            self.video,self.clips_idx = self.keys[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1 
        data = self.stackopf()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample





class ResNet3D_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel, data_path, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_path=data_path
        self.dic_nb_frame={}
        self.in_channel = in_channel
        self.img_rows=112
        self.img_cols=112
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
            video_label = pickle.load(f)
        f.close()
        #Preprocessing
        for key in dic_frame:
            video = key.split('_',1)[1].split('.',1)[0]
            self.dic_nb_frame[video]= dic_frame[key] - self.in_channel # each segment with self.in_channel frame
            #print self.dic_nb_frame
        self.dic_video_label={}
        for video in video_label:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                key = 'HandstandPushups_'+ g
            else:
                key=video
            self.dic_video_label[key]=video_label[video] 

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
                video = 'HandstandPushups_'+ g
            nb_frame = int(self.dic_nb_frame[video])
            for clip_idx in range(nb_frame):
                if clip_idx % self.in_channel ==0:
                    key = video + '-' + str(clip_idx+1)
                    self.dic_test_idx[key] = self.dic_video_label[video]

    def train_video_labeling(self):
        self.dic_video_train={}
        for video in self.train_video:
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                video = 'HandstandPushups_'+ g
            label = self.dic_video_label[video]
            nb_clips = self.dic_nb_frame[video]
            key = video +'-' + str(nb_clips)
            self.dic_video_train[key] = label 
                            
    def train(self):
        training_set = ResNet3D_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train', 
            transform = transforms.Compose([
            transforms.ToTensor(),
            ]))
        print '==> Training data :',len(training_set),' videos',training_set[1][0].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        validation_set = ResNet3D_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([self.img_rows,self.img_cols]),
            transforms.ToTensor(),
            ]))
        print '==> Validation data :',len(validation_set),' clips',validation_set[1][1].size()
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader
## Transform
class GroupRandomCrop():
    def get_params(self):
        #H = [256,224,192,168]
        #W = [256,224,192,168]
        #id1 = randint(0,len(H)-1)
        #id2 = randint(0,len(W)-1)
        self.h_crop = 112#H[id1]
        self.w_crop = 112#W[id2]
        
        self.h0 = randint(0,128-self.h_crop)
        self.w0 = randint(0,128-self.w_crop)
        
    def crop(self,img):
        img = img.resize([128,128])
        crop = img.crop([self.h0,self.w0,self.h0+self.h_crop,self.w0+self.w_crop])
        return crop  

if __name__ == '__main__':
    data_loader = ResNet3D_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        dic_path='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/resnet3d/dic/',
                                        data_path='/home/ubuntu/data/UCF101/tvl1_flow/'
                                        )
    train_loader,val_loader = data_loader.run()
    print train_loader,val_loader