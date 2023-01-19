import torch
from refhic.SRModels import RefHiCSRNet
from refhic.data import inMemoryCEPDataset,loadCEPkl
from torch_ema import ExponentialMovingAverage as EMA
from refhic.pretrain import contrastivePretrainWithoutLabel
import numpy as np

import click

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from refhic.config import checkConfig
import sys
import os

import math

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0.0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def trainModel(model,train_dataloader, optimizer, criterion, epoch, device,TBWriter=None,scheduler=None,ema=None):
    model.train()

    preds=[]

    targets = []

    iters = len(train_dataloader)
    optimizer.zero_grad()


    for i, data in enumerate(train_dataloader):
        X = torch.cat((data[0], data[1]), 1).to(device)
        y = data[2].float()[:, 0, ...].to(device)

        pred = model(X)

        loss = criterion(pred, y)

        # one graph is created here
        loss.backward()
        # graph is cleared here
        if (i + 1) % 1 == 0:
            # every 10 iterations of batches of size 10
            optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update()

        preds.append(pred.cpu())
        targets.append(y.cpu())
        if scheduler:
            scheduler.step(epoch + i/ iters)
    preds=torch.vstack(preds).flatten()
    targets=torch.vstack(targets).flatten()
    loss = criterion(preds, targets)

    if TBWriter:
        TBWriter.add_scalar("Loss/train", loss, epoch)
    return loss



def testModel(model, test_dataloaders,criterion, device,epoch,TBWriter=None,ema=None):
    model.eval()
    if ema:
        ema.store()
        ema.copy_to()
    preds=[]

    targets = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloaders):
            X = torch.cat((data[0],data[1]),1).to(device)
            y = data[2].float()[:, 0, ...].to(device)
            pred = model(X)
            preds.append(pred.cpu())
            targets.append(y.cpu())

        preds=torch.vstack(preds).flatten()
        targets=torch.vstack(targets).flatten()

        loss = criterion(preds, targets)

    if TBWriter:
        TBWriter.add_scalar("Loss/Test", loss, epoch)
    if ema:
        ema.restore()
    return loss


import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@click.command()
@click.option('--lr', type=float, default=1e-3, help='learning rate [1e-3]')
@click.option('--batchsize', type=int, default=12, help='batch size [12]')
@click.option('--epochs', type=int, default=500, help='training epochs [500]')
@click.option('--gpu', type=int, default=None, help='GPU index [auto select]')
@click.option('-w', type=int, default=200, help='submatrix size [200]')
@click.option('-n', type=int, default=10, help='sampling n samples from database; -1 for all [10]')
@click.option('--ti',type = int, default = None, help = 'use the indexed sample from the test group during training if multiple existed; values between [0,n)')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')
@click.option('--check_point',type=str,default=None,help='checkpoint')
@click.option('--useadam',type=bool,default=True,help='USE adam [True]')
@click.option('--cawr',type=bool,default=False,help ='CosineAnnealingWarmRestarts [False]')
@click.argument('traindata', type=str,required=True)
@click.argument('valdata', type=str,required=True)
@click.argument('prefix', type=str,required=True)
def train(cawr,useadam,check_point,prefix,lr,batchsize, epochs, gpu, traindata, valdata, n,ti,eval_ti,w):
    """Train RefHiC-SR for contact map enhancement

    \b
    TRAINDATA: training data folder
    VALDATA: validation data folder
    PREFIX: output prefix
    """
    parameters = {'w': w,'model':'refhicNet-CE'}
    if checkConfig():
        pass
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()

    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+"cuda:"+str(gpu))
    else:
        device = torch.device("cuda")

    im = True
    if eval_ti is not None:
        _eval_ti = {}
        for _ti in eval_ti.split(','):
            _i,_cov = _ti.split(':')
            _eval_ti[_cov] = int(_i)
        eval_ti = _eval_ti
    elif ti is not None:
        eval_ti = {'default':ti}

    if eval_ti is not None:
        print('using the following coverages for evaluations:')
        for _cov in eval_ti:
            print('        ',_cov,',',eval_ti[_cov])


    randGen = torch.Generator()
    randGen.manual_seed(42)

    if im:
        print('In memory training')
        training_data = DatasetFolder(traindata, loader=loadCEPkl, extensions='pkl')
        val_data = DatasetFolder(valdata, loader=loadCEPkl, extensions='pkl')
        train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
        X = []
        Xs = []
        Y = []
        for (x, xs, y), idx in train_dataloader:
            X.append(x[0,...].numpy())
            Xs.append(xs[0,...].numpy())
            Y.append(y[0,...].numpy())
        print(x[0,...].shape,xs[0,...].shape,y[0,...].shape,' ................')
        training_data = inMemoryCEPDataset(X, Xs, Y, samples=n)
        train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True)


        pretraining_data = inMemoryCEPDataset(X, Xs, Y, samples=n,multiTest=True)
        pretrain_dataloader = DataLoader(pretraining_data, batch_size=batchsize, shuffle=True)

        X = []
        Xs = []
        Y = []
        for (x, xs, y), idx in val_dataloader:
            X.append(x[0, ...].numpy())
            Xs.append(xs[0, ...].numpy())
            Y.append(y[0, ...].numpy())
        val_data = inMemoryCEPDataset(X, Xs, Y, samples=n)
        val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=True)


    # else:
    #     training_data = DatasetFolder(traindata, loader=loadCERandomStudyPkl, extensions='pkl')
    #     val_data = DatasetFolder(valdata, loader=loadCERandomStudyPkl, extensions='pkl')
    #     train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True, worker_init_fn=seed_worker)
    #     val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=True, worker_init_fn=seed_worker)




    # val_dataloaders = {}
    # for _k in eval_ti:
    #     val_data = inMemoryDataset(X_val,Xs_val,y_label_val,samples=None,ti=eval_ti[_k])
    #     val_dataloaders[_k] = DataLoader(val_data, batch_size=batchsize, shuffle=True,num_workers=1)

    earlyStopping = {'patience':200000000,'loss':np.inf,'wait':0,'model_state_dict':None,'epoch':0,'parameters':None}


    model=RefHiCSRNet(w=w)
    model.to(device)

    if not check_point:
        model = contrastivePretrainWithoutLabel(model, pretrain_dataloader, device=device, epochs=100)

    ema = EMA(model.parameters(), decay=0.999)
    lossfn = torch.nn.L1Loss()
    TBWriter = SummaryWriter(comment=' '+prefix)

    model.train()

    if check_point:
        _modelstate = torch.load(check_point, map_location=device)
        if 'model_state_dict' in _modelstate:
            _modelstate = _modelstate['model_state_dict']
        model.load_state_dict(_modelstate)
        print('pretrained model loaded!')

    for param in model.conv01.parameters():
        param.requires_grad = False

    for param in model.conv11.parameters():
        param.requires_grad = False

    for param in model.conv12.parameters():
        param.requires_grad = False

    for param in model.NNkv.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True)

    if useadam:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.1)

    if cawr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
    else:
        scheduler = None
        scheduler2 = CosineScheduler(int(epochs*0.9), warmup_steps=10, base_lr=lr, final_lr=1e-6)
        # schedulerLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], gamma=0.1)
    print('start train')
    for epoch in tqdm(range(0, epochs)):
        if not cawr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler2(epoch)
                TBWriter.add_scalar("LR", param_group['lr'], epoch)


        trainLoss=trainModel(model,train_dataloader, optimizer, lossfn, epoch, device,TBWriter=TBWriter,scheduler=scheduler,ema=ema)
        testLoss = testModel(model, val_dataloader, lossfn, device, epoch, TBWriter=TBWriter, ema=None)
        # if not cawr:
        #     schedulerLR.step()

        if testLoss < earlyStopping['loss'] and earlyStopping['wait']<earlyStopping['patience']:
            earlyStopping['loss'] = testLoss
            earlyStopping['epoch'] = epoch
            earlyStopping['wait'] = 0
            earlyStopping['model_state_dict'] = model.state_dict()
            earlyStopping['parameters']=parameters,
        else:
            earlyStopping['wait'] += 1
        if epoch%10==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': testLoss,
                'parameters': parameters,
            }, prefix+'_RefHiC-TAD_epoch'+str(epoch)+'.tar')

            if ema:
                ema.store()
                ema.copy_to()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': testLoss,
                    'parameters':parameters,
                }, prefix+'_RefHiC-TAD_epoch'+str(epoch)+'_ema.tar')
                ema.restore()

    print('finsh trying; best model with early stopping is epoch: ',earlyStopping['epoch'], 'loss is ',earlyStopping['loss'])
    torch.save(earlyStopping, prefix+'_RefHiC-TAD_bestModel_state.pt')

if __name__ == '__main__':
    train()
