import torch
import pandas as pd
from torch.utils.data import DataLoader
import h5py

from refhic.models import refhicNet, baselineNet,focalLoss
from torch_ema import ExponentialMovingAverage as EMA

import numpy as np
from refhic.bcooler import bcool
import click
from refhic.data import  inMemoryDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import datetime
import pickle
from refhic.pretrain import contrastivePretrain
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



def trainModel(model,train_dataloader, optimizer, criterion, epoch, device,TBWriter=None,baseline=False,scheduler=None,ema=None):
    model.train()

    preds=[]
    targets = []

    iters = len(train_dataloader)

    for batch_idx,X in enumerate(train_dataloader):
        target = X[-1].to(device)
        X[1] = X[1].to(device)
        X[2] = X[2].to(device)


        optimizer.zero_grad()
        if not baseline:
            output = model(X[1],X[2])
        else:
            output = model(X[1].to(device))

        preds.append(torch.sigmoid(output.cpu()))
        targets.append(target.cpu())
        loss = criterion(output.flatten(), target.flatten())

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if ema:
            ema.update()
        if scheduler:
            scheduler.step(epoch + batch_idx/ iters)



    preds=torch.vstack(preds).flatten()
    targets=torch.vstack(targets).flatten()


    loss = criterion(preds, targets)
    acc = torchmetrics.functional.accuracy(preds, targets.to(int))
    f1 = torchmetrics.functional.f1_score(preds, targets.to(int))
    precision, recall = torchmetrics.functional.precision_recall(preds, targets.to(int))
    positive = torch.sum(preds >= 0.5)
    negative = torch.sum(preds < 0.5)

    if TBWriter:
        TBWriter.add_scalar("Loss/train", loss, epoch)
        TBWriter.add_scalar("Accuracy/train", acc, epoch)
        TBWriter.add_scalar("F1/train", f1, epoch)
        TBWriter.add_scalar("Precision/train", precision, epoch)
        TBWriter.add_scalar("Recall/train", recall, epoch)
        TBWriter.add_scalar("positive/train", positive, epoch)
        TBWriter.add_scalar("negative/train", negative, epoch)

    return loss



def testModel(model, test_dataloaders,criterion, device,epoch,TBWriter=None,baseline=False,ema=None):
    model.eval()
    if ema:
        ema.store()
        ema.copy_to()


    with torch.no_grad():
        losses = []
        for key in test_dataloaders:
            test_dataloader = test_dataloaders[key]
            targets = []
            preds = []
            indices = []

            for X in test_dataloader:
                target = X[-1]

                target=target.to(device)


                if not baseline:
                    X[1]=X[1].to(device)
                    X[2]=X[2].to(device)
                    output = model(X[1],X[2])

                else:
                    output = model(X[1].to(device))
                output=torch.sigmoid(output)


                preds.append(output.cpu())
                targets.append(target.cpu())
                indices.append(X[0].cpu())


            preds=torch.vstack(preds).flatten()
            targets=torch.vstack(targets).flatten()

            loss = torch.mean(criterion(preds, targets))
            losses.append(loss)
            acc = torchmetrics.functional.accuracy(preds, targets.to(int))
            f1 = torchmetrics.functional.f1_score(preds, targets.to(int))
            precision, recall = torchmetrics.functional.precision_recall(preds, targets.to(int))
            positive = torch.sum(preds >= 0.5)
            negative = torch.sum(preds < 0.5)
            allPos = torch.sum(targets==1)
            allNeg = torch.sum(targets == 0)
            if TBWriter:
                TBWriter.add_scalar("Loss/test/"+key, loss, epoch)
                TBWriter.add_scalar("Accuracy/test/"+key, acc, epoch)
                TBWriter.add_scalar("F1/test/"+key, f1, epoch)
                TBWriter.add_scalar("Precision/test/"+key, precision, epoch)
                TBWriter.add_scalar("Recall/test/"+key, recall, epoch)
                TBWriter.add_scalar("positive/test/"+key, positive, epoch)
                TBWriter.add_scalar("negative/test/"+key, negative, epoch)
                TBWriter.add_scalar("allPos/test/"+key, allPos, epoch)
                TBWriter.add_scalar("allNeg/test/"+key, allNeg, epoch)

        if ema:
            ema.restore()
        return np.mean(losses)

import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@click.command()
@click.option('--lr', type=float, default=1e-3, help='learning rate [1e-3]')
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--epochs', type=int, default=1000, help='training epochs')
@click.option('--gpu', type=int, default=0, help='GPU training')
@click.option('-n', type=int, default=10, help='sampling n samples from database; -1 for all')
@click.option('--encoding_dim',type = int, default =64,help='encoding dim')
@click.option('--ti',type = int, default = None, help = 'use the indexed sample from the test group during training if multiple existed; values between [0,n)')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')
@click.option('--models',type=str,default ='refhic',help='refhic, or baseline model; [refhic]')
@click.option('--check_point',type=str,default=None,help='checkpoint')
@click.option('--pw',type=float,default=-1,help='alpha for focal loss [-1]')
@click.option('--cnn',type=bool,default=True,help='cnn encoder [True]')
@click.option('--useadam',type=bool,default=True,help='USE adam [True]')
@click.option('--lm',type=bool,default=False,help='large memory [False]')
@click.option('--cawr',type=bool,default=False,help ='CosineAnnealingWarmRestarts [False]')
@click.argument('traindata', type=str,required=True)
@click.argument('prefix', type=str,required=True)
def train2(cawr,lm,useadam,cnn,pw,check_point,prefix,lr,batchsize, epochs, gpu, traindata, n,ti,encoding_dim,models,eval_ti):
    """Train RefHiC for TAD annotation"""

    parameters={'cnn':cnn,'encoding_dim':encoding_dim,'model':'refhicNet-TAD'}
    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+"cuda:"+str(gpu))
    else:
        device = torch.device("cpu")
    if lm:
        occccccc = torch.zeros((256,1024,18000)).to(device)

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




    with open(traindata, 'rb') as handle:
        dataParams,\
        X_train, Xs_train, y_label_train, \
        X_test, Xs_test, y_label_test, \
        X_val, Xs_val, y_label_val = pickle.load(handle)

    for key in dataParams:
        parameters[key]=dataParams[key]

    if eval_ti is None:
        eval_ti = {}
        for _i in range(X_train[0].shape[0]):
            eval_ti['sample'+str(_i)] = _i
    print('eval_ti',eval_ti)
    print('#train:',len(y_label_train))
    print('#test:', len(y_label_test))
    print('#validation:', len(y_label_val))
    print('#training cases',len(y_label_train))
    if n == -1:
        n = None

    training_data = inMemoryDataset(X_train,Xs_train,y_label_train,samples=n,ti=ti)
    pretrain_data = inMemoryDataset(X_train,Xs_train,y_label_train,samples=n,ti=ti,multiTest=True)

    randGen = torch.Generator()
    randGen.manual_seed(42)
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True,num_workers=1,worker_init_fn = seed_worker)
    pretrain_dataloader = DataLoader(pretrain_data, batch_size=512, shuffle=True, num_workers=1,
                                  worker_init_fn=seed_worker)

    val_dataloaders = {}
    for _k in eval_ti:
        val_data = inMemoryDataset(X_val,Xs_val,y_label_val,samples=None,ti=eval_ti[_k])
        val_dataloaders[_k] = DataLoader(val_data, batch_size=batchsize, shuffle=True,num_workers=1)

    if 'refhic' in models.lower():
        earlyStopping = {'patience':200000000,'loss':np.inf,'wait':0,'model_state_dict':None,'epoch':0,'parameters':None}
        model = refhicNet(parameters['featureDim'], encoding_dim=encoding_dim, CNNencoder=cnn,
                          win=2 * parameters['w'] + 1,
                          classes=parameters['classes']).to(device)

        ema = EMA(model.parameters(), decay=0.999)

        criterion = focalLoss(alpha=pw, gamma=2, adaptive=True)

        ### pretrain
        if check_point is None:
            model=contrastivePretrain(model,pretrain_dataloader,0.001,20,device)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, prefix + '_refhicNet-TAD_pretrain.tar')
            pass
        ### end of pretrain


        TBWriter = SummaryWriter(comment=' '+prefix)

        model.train()

        if check_point:
            _modelstate = torch.load(check_point, map_location='cuda:' + str(gpu))
            if 'model_state_dict' in _modelstate:
                _modelstate = _modelstate['model_state_dict']
            model.load_state_dict(_modelstate)



        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True)
        if useadam:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.1)

        if cawr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
        else:
            scheduler = None
            scheduler2 = CosineScheduler(int(epochs*0.9), warmup_steps=5, base_lr=lr, final_lr=1e-6)


        for epoch in tqdm(range(0, epochs)):
            if not cawr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler2(epoch)

            trainLoss=trainModel(model,train_dataloader, optimizer, criterion, epoch, device,TBWriter=TBWriter,scheduler=scheduler,ema=ema)
            testLoss=testModel(model,val_dataloaders, criterion,device,epoch,TBWriter=TBWriter,ema=None)

            # schedulerLR.step()


            if testLoss < earlyStopping['loss'] and earlyStopping['wait']<earlyStopping['patience']:
                earlyStopping['loss'] = testLoss
                earlyStopping['epoch'] = epoch
                earlyStopping['wait'] = 0
                earlyStopping['model_state_dict'] = model.state_dict()
                earlyStopping['parameters']=parameters
            else:
                earlyStopping['wait'] += 1
            if epoch%10==0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': testLoss,
                    'parameters': parameters,
                }, prefix+'_refhicNet-TAD_epoch'+str(epoch)+'.tar')

                if ema:
                    ema.store()
                    ema.copy_to()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': testLoss,
                        'parameters': parameters,
                    }, prefix+'_refhicNet-TAD_epoch'+str(epoch)+'_ema.tar')
                    ema.restore()

        print('finsh trying; best model with early stopping is epoch: ',earlyStopping['epoch'], 'loss is ',earlyStopping['loss'])
        torch.save(earlyStopping, prefix+'_refhicNet-TAD_bestModel_state.pt')


    if 'baseline' in models.lower():
        # baseline model
        parameters['model']='baselineNet-TAD'
        baselineModel = baselineNet(parameters['featureDim'],encoding_dim=encoding_dim,win=2*parameters['w']+1).to(device)
        ema = EMA(baselineModel.parameters(), decay=0.999)
        TBWriterB = SummaryWriter(comment=' baseline '+prefix)
        baselineModel.train()
        #criterion = torch.nn.BCEWithLogitsLoss()
        criterion = focalLoss(alpha=pw, gamma=2,adaptive=True)
        # optimizer = torch.optim.Adam(baselineModel.parameters(), lr=lr, eps=1e-8, amsgrad=True)
        optimizer = torch.optim.AdamW(baselineModel.parameters(), lr=lr, weight_decay=0.1)
        scheduler = None#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
        scheduler2 = CosineScheduler(int(epochs * 0.95), warmup_steps=0, base_lr=lr, final_lr=1e-6)

        earlyStopping = {'patience': 500000, 'loss': np.inf, 'wait': 0, 'model_state_dict': None, 'epoch': 0,'parameters':None}

        for epoch in tqdm(range(epochs)):
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler2(epoch)
            trainModel(baselineModel,train_dataloader, optimizer, criterion, epoch, device,TBWriter=TBWriterB,baseline=True,scheduler=scheduler,ema=ema)
            testLoss=testModel(baselineModel,val_dataloaders, criterion,device,epoch,TBWriter=TBWriterB,baseline=True)

            if testLoss < earlyStopping['loss'] and earlyStopping['wait'] < earlyStopping['patience']:
                earlyStopping['loss'] = testLoss
                earlyStopping['epoch'] = epoch
                earlyStopping['wait'] = 0
                earlyStopping['parameters']=parameters
                earlyStopping['state'] = baselineModel.state_dict()
            else:
                earlyStopping['wait'] += 1
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': baselineModel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': testLoss,
                    'parameters':parameters,
                }, prefix + '_baseline-TAD_epoch' + str(epoch) + '.tar')
                if ema:
                    ema.store()
                    ema.copy_to()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': baselineModel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': testLoss,
                        'parameters': parameters,
                    }, prefix + '_baseline-TAD_epoch' + str(epoch) + '_ema.tar')
                    ema.restore()


        print('finsh trying; best model with early stopping is epoch: ', earlyStopping['epoch'], 'loss is ',
              earlyStopping['loss'])
        torch.save(earlyStopping, prefix+'_baseline-TAD_bestModel_state.pt')







if __name__ == '__main__':
    train2()
