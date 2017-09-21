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

class UCF_training_dataset(Dataset):  
    def __init__(self, dic, rgb_root, opf_root, transform=None):
        #print('==> get total frame numbers of each video')
        frame_count={}
        for key in dic:
            videoname = key.split('/')[2]
            frame = key.split('_')[-1].split('.',1)[0]
            if videoname not in frame_count.keys():
                frame_count[videoname] = int(frame)
            else:
                if int(frame) > frame_count[videoname]:
                    frame_count[videoname] = int(frame)
        #print frame_count
        dic_stack={}
        for key in dic:
            videoname = key.split('/')[2]
            frame = key.split('_')[-1].split('.',1)[0]
            if int(frame) < frame_count[videoname]-9:
                dic_stack[key] = dic[key]
 
        self.keys = dic_stack.keys()
        self.values = dic_stack.values()
        self.rgb_root = rgb_root
        self.opf_root = opf_root
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        key = self.keys[idx]
        #rgb
        #print '\n',key
        if key.split('/',1)[0] == 'HandStandPushups':
            key = 'HandstandPushups/'+key.split('/',1)[1]

        img = Image.open(self.rgb_root + key)
        rgb_data = self.transform(img)
        img.close()
        t2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        rgb_data=t2(rgb_data)
        #opf
        v,n1,n2,n3,stack_idx = key.split('/')[-1].split('.',1)[0].split('_',4)
        classname = n1+ '_'+ n2+'_'+ n3
        #stack_idx= str((int(stack_idx)-1)*10 +1)
        opf_data = stackopf(classname,stack_idx,self.opf_root,self.transform)
    
        label = int(self.values[idx])-1
        sample=(rgb_data,opf_data,label)
        return sample

class UCF_testing_dataset(Dataset):  
    def __init__(self, ucf_list, rgb_root, opf_root, transform=None):

        frame_count={}
        for key in ucf_list:
            name,label = key.split('[@]')
            videoname,idx = name.split('-')

            if videoname not in frame_count.keys():
                frame_count[videoname] = int(idx)
            else:
                if int(idx) > frame_count[videoname]:
                    frame_count[videoname] = int(idx)
        #print frame_count
        dic_stack=[]
        for key in ucf_list:
            name,label = key.split('[@]')
            videoname,idx = name.split('-')

            if int(idx) < frame_count[videoname]-9:
                dic_stack.append(key)
 
 
        self.ucf_list = dic_stack
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
        rgb_data = self.transform(img)
        t2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        rgb_data=t2(rgb_data)


        img.close()

        #optical flow
        opf_data = stackopf(video_name,idx,self.opf_root,self.transform)

        sample=(video_name,rgb_data,opf_data,label)
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
        idx = str((i-1)+j) # for overlap dataset
        #idx = str(10*(i-1)+j) # for nonoverlap dataset
        frame_idx = 'frame'+ idx.zfill(6)
        h_image = u +'/' + frame_idx +'.jpg'
        v_image = v +'/' + frame_idx +'.jpg'
        
        imgH=(Image.open(h_image))
        imgV=(Image.open(v_image))

        H = transform(imgH)
        V = transform(imgV)
        flow[2*(j-1),:,:] = H
        flow[2*(j-1)+1,:,:] = V      
        imgH.close()
        imgV.close() 

    #opf_data = torch.from_numpy(flow).float().sub_(127.353346189).div_(14.971742063)
    opf_data = torch.from_numpy(flow).float().div_(255)  
    return opf_data

# Test data loader
if __name__ == '__main__':
    import numpy as np
    rgb_root='/home/ubuntu/data/UCF101/spatial/'
    opf_root='/home/ubuntu/data/UCF101/tvl1_flow/'
    dic_path='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/dictionary/spatial/'


    with open(dic_path+'/dic_training.pickle','rb') as f:
        dic_training=pickle.load(f)
    f.close()

    with open(dic_path+'/dic_test25.pickle','rb') as f:
        dic_testing=pickle.load(f)
    f.close()

    training_set = UCF_training_dataset(dic=dic_training, rgb_root='/home/ubuntu/data/UCF101/spatial_no_sampled/', opf_root=opf_root, transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            ]))
    
    validation_set = UCF_testing_dataset(ucf_list=dic_testing, rgb_root='/home/ubuntu/data/UCF101/spatial_no_sampled/', opf_root=opf_root,transform = transforms.Compose([
            transforms.CenterCrop(224),
            #transforms.ToTensor(),
            ]))

    print training_set[1]