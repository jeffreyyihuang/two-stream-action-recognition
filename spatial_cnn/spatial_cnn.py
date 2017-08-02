import numpy as np
import pickle,os
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
import shutil
import argparse
import lr_scheduler

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Optimizer

torch.cuda.set_device(0)
print '\nCurrent Device',torch.cuda.current_device(),' Available:',torch.cuda.is_available()

# Default value
parser = argparse.ArgumentParser(description='PyTorch UCF101 spatial stream training')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()

    # Hyper Parameters
    EPOCH = args.epochs         
    BATCH_SIZE = args.batch_size
    LR = args.lr 
    print '==> Create Model'
    #Create Model
    Model = models.resnet101(pretrained=True)
      #Replace fc1000 with fc101
    num_ftrs = Model.fc.in_features
    Model.fc = nn.Linear(num_ftrs, 101)
      #convert model to gpu
    Model = Model.cuda()

    #Loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(Model.parameters(), LR, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0,verbose=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))
    
    print '==> Preparing training and validation data'
    #load dictionary
    dic_training, dic_testing = load_dic()

    # change DIC -> LIST format due to time cost
    L_train_keys = dic_training.keys()
    L_train_values = dic_training.values()

    L_test_keys = dic_testing.keys()
    L_test_values = dic_testing.values()
       
    #Build training & testing set
    training_set = UCF101_rgb_data(keys=L_train_keys, values=L_train_values, root_dir='/store/ucf101/spatial/',transform = transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
    
    testing_set = UCF101_rgb_data(keys=L_test_keys, values=L_test_values, root_dir='/store/ucf101/spatial/',transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
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
    
    cudnn.benchmark = True
    # Evaluation mode
    if args.evaluate:
        validate(test_loader, Model, criterion,L_test_keys, BATCH_SIZE)
        return

    print '==> Start training'
    #Training & testing
    for epoch in range(args.start_epoch, EPOCH):

        # train for one epoch
        print(' Epoch:[{0}/{1}][training stage]'.format(epoch, EPOCH))
        train(train_loader, Model, criterion, optimizer, epoch)

        # evaluate on validation set
        print(' Epoch:[{0}/{1}][validation stage]'.format(epoch, EPOCH))
        prec1, val_loss, dic_video_level_preds = validate(test_loader, Model, criterion, 0, L_test_keys,BATCH_SIZE)
        
        #Call lr_scheduler
        scheduler.step(val_loss)

        #Calculate Video level acc
        top1,top5 = video_level_acc(dic_video_level_preds)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet101',
            'state_dict': Model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,dic_video_level_preds)


def load_dic():

    dic_path = '../dictionary/spatial/'

    with open(dic_path+'dic_training.pickle','rb') as f:
        dic_training=pickle.load(f)
    f.close()

    with open(dic_path+'dic_testing.pickle','rb') as f:
        dic_testing=pickle.load(f)
    f.close()

    return dic_training,dic_testing

# My Dataset
class UCF101_rgb_data(Dataset):  
    def __init__(self, keys,values, root_dir, transform=None):
 
        self.keys = keys
        self.values = values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        frame = self.keys[idx] 
        img = Image.open(self.root_dir + frame)
        label = self.values[idx]
        label = int(label)-1

        if self.transform:
            transformed_img = self.transform(img)
            sample = (transformed_img,label)
        else:
            sample = (img,label)
                 
        img.close()
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
        # measure data loading time
        data_time.update(time.time() - end)
        
        label = label.cuda(async=True)
        input_var = Variable(data).cuda()
        target_var = Variable(label).cuda()

        # compute output
        output = Model(input_var)
        loss = criterion(output, target_var)

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
        ii = i+1
        if ((ii % (len(train_loader)/10)) == 0):

            # Show info on console
    	    result = ('Epoch: [{0}],Training[{1}/{2}]\t'
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
            
def validate(test_loader, Model, criterion, L_test_keys, BATCH_SIZE):
    
    dic_video_level_preds = {}
    L_video=[]
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    Model.eval()

    end = time.time()
    progress = tqdm(test_loader)
    for i, (data,label) in enumerate(progress):
        
        
        l = len(label) 
        key = L_test_keys[i*BATCH_SIZE:i*BATCH_SIZE+l]
        
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

        #Calculate video level prediction  dic[video_name] = prediction score
        preds = output.data.cpu().numpy()
        nb_data = preds.shape[0]

        for i in range(nb_data):
            name = key[i].split('/')[2].split('_',1)[1]
            if name not in L_video:
                dic_video_level_preds[name]= preds[i,:]
                L_video.append(name)
            else:
                dic_video_level_preds[name] += preds[i,:]
    
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        ii=i+1
        if ii % (len(test_loader)/5) == 0:
            # Show info on console
            result = (' Testing[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
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
            

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, losses.avg, dic_video_level_preds

def video_level_acc(dic_video_level_preds):
    
    with open('../dictionary/dic_video_label.pickle','rb') as f:
        dic_video_label = pickle.load(f)
    f.close()
    
    correct = 0
    video_level_preds = np.zeros((len(dic_video_level_preds),101))
    video_level_labels = np.zeros(len(dic_video_level_preds))
    ii=0
    for line in sorted(dic_video_level_preds.keys()):
        preds = dic_video_level_preds[line]
        label = int(dic_video_label[line])-1
        
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
   
    #record as csv file
    filename = 'record/video_level_acc.csv'
    column_names = ['Prec@1','Prec@5']
    df =pd.DataFrame({'Prec@1':[top1],'Prec@5':[top5]})

    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)
    
    print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
    return top1,top5

#other function 
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

def save_checkpoint(state, is_best, dic_video_level_preds, filename='../save/spatial_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../save/spatial_model_best.pth.tar')
        with open('../save/dic_spaital_video_level_preds.pickle','wb') as f:
            pickle.dump(dic_video_level_preds,f)
        f.close()

if __name__ == '__main__':
    main()
