import numpy as np
import pickle,os
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
import shutil
import lr_scheduler

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#
import my_resnet_20channel as mR20


torch.cuda.set_device(1)
print'\nCurrent Device:', torch.cuda.current_device(),' Available:',torch.cuda.is_available()

def load_dic():

    dic_path = '/home/jeffrey/pytorch/ucf101_two_stream/motion_cnn/dictionary/'

    with open(dic_path+'dic_training.pickle','rb') as f:
        dic_training=pickle.load(f)
    f.close()

    with open(dic_path+'dic_testing.pickle','rb') as f:
        dic_testing=pickle.load(f)
    f.close()

    return dic_training,dic_testing

def stackopf(classname,stack_idx,opf_img_path,transform):

    #opf_img_path = '/home/jeffrey/data/tvl1_flow/'
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
    return flow

class UCF101_opf_data(Dataset):  
    def __init__(self, keys,values, root_dir, transform):

        self.keys = keys
        self.values = values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        classname,stack_idx = key.split('-')
        
        stack_opf_image = stackopf(classname,stack_idx,self.root_dir,self.transform)
        
        label = self.values[idx]
        label = int(label)-1

        sample = (torch.from_numpy(stack_opf_image).float(),label)
    
        return sample

def train(train_loader, Model, criterion, optimizer, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    #switch to train mode
    Model.train()
    
    end = time.time()
    progress = tqdm(train_loader)
    for i, (data,label) in enumerate(progress):
        #if i >20 :break

        # Change data to Normal distribution
        #Mean: 127.353346189, std : 14.971742063

        data = data.sub_(127.353346189).div_(14.971742063)

        # measure data loading time
        data_time.update(time.time() - end)
        
        label = label.cuda(async=True)
        data_var = Variable(data).cuda()
        label_var = Variable(label).cuda()

        # compute output
        output = Model(data_var)
        loss = criterion(output, label_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #show    
        ii=i+1
        if ((ii % (len(train_loader)/1000)) == 0):
    	    result = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            print result
            print '----'*40

            # Save the information to training.csv file
            filename = 'record/training.csv'
            column_names = [
                            'Epoch',
                            'Progress',
                            'Batch Time',
                            'Data Time',
                            'Loss',
                            'Prec@1',
                            'Prec@5'
                            ]

            prog = str(round((float(i)/float(len(train_loader))),2)*100)+' %'

            df =pd.DataFrame.from_dict({
                'Epoch':[epoch],
                'Progress' : [prog],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[losses.avg],
                'Prec@1':[top1.avg],
                'Prec@5':[top5.avg]
                })

            if not os.path.isfile(filename):
                df.to_csv(filename,index=False,columns=column_names)
            else: # else it exists so append without writing the header
                df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)
            
def validate(test_loader, Model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    Model.eval()

    end = time.time()
    progress = tqdm(test_loader)
    for i, (data,label) in enumerate(progress):
        
        #if i >20:break
        # Change data to Normal distribution
        data = data.sub_(127.353346189).div_(14.971742063)


        label = label.cuda(async=True)
        data_var = Variable(data, volatile=True).cuda(async=True)
        label_var = Variable(label, volatile=True).cuda(async=True)

        # compute output
        output = Model(data_var)
        loss = criterion(output, label_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ii = i+1
        if ii % (len(test_loader)/10) == 0:
            # Show info on console
            result = ('Epoch: [{0}], Testing[{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch,
                       i, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            print result
            print '----'*40

            # Save the information to testing.csv file
            filename = 'record/testing.csv'
            column_names = [
                            'Epoch',
                            'Progress',
                            'Batch Time',
                            'Loss',
                            'Prec@1',
                            'Prec@5'
                            ]

            prog = str(round((float(i)/float(len(test_loader))),2)*100)+' %'
            
            df =pd.DataFrame({
                'Epoch':[epoch],
                'Progress' : [prog],
                'Batch Time':[round(batch_time.avg,2)],
                'Loss':[losses.avg],
                'Prec@1':[top1.avg],
                'Prec@5':[top5.avg]
                })

            if not os.path.isfile(filename):
                df.to_csv(filename,index=False,columns=column_names)
            else: # else it exists so append without writing the header
                df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)            

    print(' * Prec@1: {top1.avg:.3f} Prec@5: {top5.avg:.3f} Loss: {loss.avg:.4f} '.format(top1=top1, top5=top5, loss=losses))
    return top1.avg, losses.avg

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

def save_checkpoint(state, is_best, filename='../save/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../save/model_best.pth.tar')

def w3_to_w20(conv1_weight):
    S = 0
    for i in range(3):
        S += conv1_weight[:,i,:,:]

    avg = S/3.
    
    new_conv1_weight = torch.FloatTensor(64,20,7,7)
    for i in range(20):
        new_conv1_weight[:,i,:,:] = avg
        
    return new_conv1_weight

def main():
	    # Hyper Parameters
    EPOCH = 50            # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 64
    LR = 1e-3          # learning rate
    nb_classes = 101
    
    #load dictionary
    dic_training, dic_testing = load_dic()

    #change DIC into LIST format due to time cost
    L_train_keys = dic_training.keys()
    L_train_values = dic_training.values()

    L_test_keys = dic_testing.keys()
    L_test_values = dic_testing.values()

    #Build training & testing set
    training_set = UCF101_opf_data(keys = L_train_keys, values=L_train_values , root_dir='/home/jeffrey/data/tvl1_flow/',transform=transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.Normalize((0.5,), (1.0,))
                ]))   
    testing_set = UCF101_opf_data(keys=L_test_keys, values=L_test_values, root_dir='/home/jeffrey/data/tvl1_flow/',transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                #transforms.Normalize((0.5,), (1.0,))
                ]))
    #Data Loader
    train_loader = DataLoader(
        dataset=training_set, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
        )
    test_loader = DataLoader(
        dataset=testing_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=8
        )
    
    # Build 3 channel ResNet101 with pre-trained weight
    ResNet101 = models.resnet101(pretrained=True)
    
    #Build 20 channel ResNet101
    Model = mR20.resnet101()
    
    #1.Get the weight of first convolution layer (torch.FloatTensor of size 64x3x7x7).
    conv1_weight = ResNet101.state_dict()['conv1.weight']
    dic = ResNet101.state_dict()

    #3.Average across rgb channel and replicate this average by the channel number of target network(20 in this case)
    conv1_weight20 = w3_to_w20(conv1_weight)
    dic['conv1.weight'] = conv1_weight20
    Model.load_state_dict(dic)
    
    
    #Replace fc1000 with fc101
    num_ftrs = Model.fc.in_features
    Model.fc = nn.Linear(num_ftrs, nb_classes)
    
    #convert model to gpu
    Model = Model.cuda()
    

    #Loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer and lr_scheduler
    optimizer = torch.optim.SGD(Model.parameters(), LR,momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0,verbose=True)


    cudnn.benchmark = True
    
        #Training & testing
    best_prec1 = 0
    for epoch in range(1,EPOCH+1):

        print('****'*40)
        print('Epoch:[{0}/{1}][training stage]'.format(epoch, EPOCH))
        print('****'*40)
        # train for one epoch
        train(train_loader, Model, criterion, optimizer, epoch)

        # evaluate on validation set
        print('****'*40)
        print('Epoch:[{0}/{1}][validation stage]'.format(epoch, EPOCH))
        print('****'*40)

        prec1, val_loss = validate(test_loader, Model, criterion, epoch)
        
        #Call lr_scheduler
        scheduler.step(val_loss)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch,
            'arch': 'resnet101',
            'state_dict': Model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
	main()
