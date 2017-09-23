import pickle,os
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
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
from torch.optim import Optimizer

class UCF_training_set(Dataset):  
    def __init__(self, dic, root_dir, transform=None):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        frame = self.keys[idx]
        if frame.split('/',1)[0] == 'HandStandPushups':
            frame = 'HandstandPushups/'+frame.split('/',1)[1]
        img = Image.open(self.root_dir + frame)
        label = self.values[idx]
        label = int(label)-1

        transformed_img = self.transform(img)
        

        v,classname,g,c,idx = frame.split('/')[-1].split('.',1)[0].split('_',4)
        
        key = classname+'/'+classname+'_'+g+'_'+c+'/'+'[@]'+str(idx).zfill(6)
        img.close()

        sample = (key,transformed_img,label)
        return sample

class UCF_testing_set(Dataset):  
    def __init__(self, ucf_list, root_dir, transform=None):
 
        self.ucf_list = ucf_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ucf_list)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        key,label = self.ucf_list[idx].split('[@]')
        video_name,idx = key.split('-')
	

        #open image
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandStandPushups_'+g
            path = self.root_dir + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
        else:
            path = self.root_dir + video_name.split('_')[0]+'/separated_images/v_'+video_name+'/v_'+video_name+'_'
         
        img = Image.open(path +str(idx)+'.jpg')
        label = int(label)-1

        if self.transform:
            transformed_img = self.transform(img)
        img.close()

        name = video_name.split('_',1)[0]+'/'+video_name+'/'+'[@]'+str(idx).zfill(6)


        sample = (name,transformed_img, label)
        return sample

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