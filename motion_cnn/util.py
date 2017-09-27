import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Optimizer

# Dataset
class UCF101_opf_training_set(Dataset):  
    def __init__(self, dic_video_training, dic_nb_frame, root_dir, transform):

        self.dic_nb_stack = {}
        for key in dic_nb_frame:
            videoname=key.split('.',1)[0].split('_',1)[1]
            self.dic_nb_stack[videoname]=dic_nb_frame[key]-9



        self.keys = dic_video_training.keys()
        self.values = dic_video_training.values()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        videoname = self.keys[idx]
        n,g = videoname.split('_',1)
        if n == 'HandStandPushups':
            videoname = 'HandstandPushups_'+g



        nb_frame = int(self.dic_nb_stack[videoname])
        stack_idx = randint(1,nb_frame)

        stack_opf_image = stackopf(videoname,stack_idx,self.root_dir,self.transform)
        
        label = self.values[idx]
        label = int(label)-1

        sample = (stack_opf_image,label)
    
        return sample

class UCF101_opf_testing_set(Dataset):  
    def __init__(self, List, root_dir, transform):

        self.List = List
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.List)

    def __getitem__(self, idx):
    
        key = self.List[idx]
        name,label = key.split('[@]')
        classname,stack_idx = name.split('-')
        
        stack_opf_image = stackopf(classname,stack_idx,self.root_dir,self.transform)
        
        label = int(label)-1
        sample = (name,stack_opf_image,label)
        return sample

def stackopf(classname,stack_idx,opf_img_path,transform):
    #opf_img_path = '/home/jeffrey/data/tvl1_flow/'
    name = 'v_'+classname
    u = opf_img_path+ 'u/' + name
    v = opf_img_path+ 'v/'+ name
    
    flow = np.zeros((20,224,224))
    i = int(stack_idx)
    r_crop = Random_Crop()
    r_crop.get_crop_size()

    for j in range(1,10+1):
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        idx = str((i-1)+j) # for overlap dataset
        #idx = str(10*(i-1)+j) # for nonoverlap dataset
        frame_idx = 'frame'+ idx.zfill(6)
        h_image = u +'/' + frame_idx +'.jpg'
        v_image = v +'/' + frame_idx +'.jpg'
        
        imgH=(Image.open(h_image))
        imgV=(Image.open(v_image))
        if transform:
            H = r_crop.crop_and_resize(transform(imgH))
            V = r_crop.crop_and_resize(transform(imgV))
            
        else:
            H = imgH.resize([224,224])
            V = imgV.resize([224,224])
        
        flow[2*(j-1),:,:] = H
        flow[2*(j-1)+1,:,:] = V      
        imgH.close()
        imgV.close()  
    return torch.from_numpy(flow).float().div(255)
#customize random cropping
class Random_Crop():
    def get_crop_size(self):
        H = [256,224,192,168]
        W = [256,224,192,168]
        id1 = randint(0,len(H)-1)
        id2 = randint(0,len(W)-1)
        self.h_crop = H[id1]
        self.w_crop = W[id2]
        
        self.h0 = randint(0,256-self.h_crop)
        self.w0 = randint(0,256-self.w_crop)
        

    def crop_and_resize(self,img):
        crop = img.crop([self.h0,self.w0,self.h_crop,self.w_crop])
        resize = crop.resize([224,224])
        return resize    
    
# other util
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='record/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'record/model_best.pth.tar')
'''
def set_channel(dic, channel):
    dic_stack={}
    for key in dic:
        frame_idx = int(key.split('/')[-1].split('.',1)[0])
        if frame_idx % channel == 0:
            dic_stack[key] = dic[key]

    return dic_stack
'''
def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names) 

# Test data loader
if __name__ == '__main__':
    import numpy as np
    data_path='/home/ubuntu/data/JHMDB/pose_estimation/pose_estimation/'
    dic_path='/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/'


    with open(dic_path+'/dic_pose_train.pickle','rb') as f:
        dic_training=pickle.load(f)
    f.close()

    with open(dic_path+'/dic_pose_test.pickle','rb') as f:
        dic_testing=pickle.load(f)
    f.close()

    training_set = JHMDB_Pose_heatmap_data_set(dic=dic_training, root_dir=data_path, nb_per_stack=10, transform = transforms.Compose([
            transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            ]))
    
    validation_set = JHMDB_Pose_heatmap_data_set(dic=dic_testing, root_dir=data_path, nb_per_stack=10,transform = transforms.Compose([
            transforms.CenterCrop(224),
            #transforms.ToTensor(),
            ]))
    print type(training_set[1][1][1,:,:].numpy())
    a = (training_set[1][1][1,:,:].numpy())
    with open('test.pickle','wb') as f:
        pickle.dump(a,f)
    f.close()
