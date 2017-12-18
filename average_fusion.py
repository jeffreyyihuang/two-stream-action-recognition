from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader






if __name__ == '__main__':

    rgb_preds='record/spatial/spatial_video_preds.pickle'
    opf_preds = 'record/motion/motion_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                    path='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
                                    ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
                                    ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),101))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    for name in sorted(rgb.keys()):   
        r = rgb[name]
        o = opf[name]
        #preds=p+r
        label = int(test_video[name])-1
                    
        video_level_preds[ii,:] = (r+o)
        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        
    top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                                
    print top1,top5
