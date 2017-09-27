import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

class UCF_testing_dataset(Dataset):  
    def __init__(self, ucf_list, rgb_root, opf_root, transform=None):

        self.ucf_list = ucf_list
        self.rgb_root = rgb_root
        self.opf_root = opf_root
        self.transform = transform

    def __len__(self):
        return len(self.ucf_list)

    def __getitem__(self, idx):
        key,label = self.ucf_list[idx].split('[@]')
        video_name,idx = key.split('-')
        label = int(label)-1  

        #rgb image
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandStandPushups_'+g
            path = self.rgb_root + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
        else:
            path = self.rgb_root + video_name.split('_')[0]+'/separated_images/v_'+video_name+'/v_'+video_name+'_'
         
        img = Image.open(path +str(idx)+'.jpg')
        rgb_data = (self.transform(img)).resize([224,224])
        t2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        rgb_data=t2(rgb_data)
        img.close()

        #optical flow
        opf_data = stackopf(video_name,idx,self.opf_root,self.transform)

        sample=(video_name, rgb_data, opf_data, label)
        return sample

def stackopf(classname,stack_idx,opf_img_path,transform):
    #opf_img_path = '/home/jeffrey/data/tvl1_flow/'
    n,g = classname.split('_',1)
    if n == 'HandStandPushups':
        classname = 'HandstandPushups_'+ g

    name = 'v_'+classname
    u = opf_img_path+ 'u/' + name
    v = opf_img_path+ 'v/'+ name
    
    flow = np.zeros((20,224,224))
    i = int(stack_idx)

    for j in range(1,10+1):
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        idx = str((i-1)+j) 
        #idx = str(10*(i-1)+j) # for nonoverlap dataset
        frame_idx = 'frame'+ idx.zfill(6)
        h_image = u +'/' + frame_idx +'.jpg'
        v_image = v +'/' + frame_idx +'.jpg'
        
        imgH=(Image.open(h_image))
        imgV=(Image.open(v_image))

        H = (transform(imgH)).resize([224,224])
        V = (transform(imgV)).resize([224,224])

        flow[2*(j-1),:,:] = H
        flow[2*(j-1)+1,:,:] = V      
        imgH.close()
        imgV.close() 

    #opf_data = torch.from_numpy(flow).float().sub_(127.353346189).div_(14.971742063)
    opf_data = torch.from_numpy(flow).float().div_(255)  
    return opf_data
