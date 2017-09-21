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
              'rgb loss {rgb_loss} '
              'rgb Prec@1 {rgb_top1} '
              'rgb Prec@5 {rgb_top5}\n'
              'opf loss {opf_loss} '
              'opf Prec@1 {opf_top1} '
              'opf Prec@5 {opf_top5}\n'
              .format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], rgb_loss=info['rgb loss'], rgb_top1=info['rgb Prec@1'], rgb_top5=info['rgb Prec@5'],
               opf_loss=info['opf loss'], opf_top1=info['opf Prec@1'], opf_top5=info['opf Prec@5']))      
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','rgb loss','rgb Prec@1','rgb Prec@5','opf loss','opf Prec@1','opf Prec@5']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Spatial Loss {rgb_loss} '
              'rgb Prec@1 {rgb_top1} '
              'rgb Prec@5 {rgb_top5} \n'
              'Motion Loss {opf_loss} '
              'opf Prec@1 {opf_top1} '
              'opf Prec@5 {opf_top5} \n'
              'fuse Prec@1 {fuse_top1} '
              'fuse Prec@5 {fuse_top5} \n'
              .format( batch_time=info['Batch Time'],
               rgb_loss=info['rgb loss'], rgb_top1=info['rgb Prec@1'], rgb_top5=info['rgb Prec@5'],
               opf_loss=info['opf loss'], opf_top1=info['opf Prec@1'], opf_top5=info['opf Prec@5'],
               fuse_top1=info['fuse Prec@1'], fuse_top5=info['fuse Prec@5']))
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','rgb loss','rgb Prec@1','rgb Prec@5','opf loss','opf Prec@1','opf Prec@5','fuse Prec@1','fuse Prec@5']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names) 


