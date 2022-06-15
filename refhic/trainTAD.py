import torch
import pandas as pd
from torch.utils.data import DataLoader
from refhic.models import refhicNet
from torch_ema import ExponentialMovingAverage as EMA

import numpy as np
from refhic.bcooler import bcool
import click
from refhic.data import inMemoryDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import pickle
from refhic.config import checkConfig
import sys
from refhic.config import referenceMeta


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
        target = X[-1][...,0,:].to(device)
        X[1] = X[1].to(device)
        X[2] = X[2].to(device)

        optimizer.zero_grad()
        output = model(X[1],X[2])


        loss = criterion(output, target)
        preds.append(output.cpu())
        targets.append(target.cpu())


        loss.backward()

        optimizer.step()
        if ema:
            ema.update()
        if scheduler:
            scheduler.step(epoch + batch_idx/ iters)



    preds=torch.vstack(preds).flatten()
    targets=torch.vstack(targets).flatten()

    loss = criterion(preds, targets)


    if TBWriter:
        TBWriter.add_scalar("Loss/train", loss, epoch)


    return loss



def testModel(model, test_dataloaders,criterion, device,epoch,printData=False,TBWriter=None,ema=None):
    model.eval()
    if ema:
        ema.store()
        ema.copy_to()
    if printData:
        print('printData',printData)
        missclassfied = {}
        _filename = "missclassfied"
        _suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S.pkl")
        missclassfied['filename'] = "_".join([_filename, _suffix])
        missclassfied['X'] = []
        missclassfied['Xs'] = []
        missclassfied['target'] = []
        missclassfied['pred'] = []

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

                X[1]=X[1].to(device)
                X[2]=X[2].to(device)
                output = model(X[1],X[2])


                if printData:
                    failed= ((output>0.5)*1!=target).cpu().numpy().flatten()
                    print('#failed',np.sum(failed))
                    missclassfied['pred'].append(output.cpu().numpy().flatten()[failed])
                    missclassfied['target'].append(target.cpu().numpy().flatten()[failed])
                    missclassfied['X'].append(X[1][failed,...].cpu().numpy())
                    missclassfied['Xs'].append(X[2][failed,...].cpu().numpy())

                preds.append(output.cpu())
                targets.append(target.cpu())
                indices.append(X[0].cpu())


            preds=torch.vstack(preds).flatten()
            targets=torch.vstack(targets).flatten()

            loss = torch.mean(criterion(preds, targets))
            losses.append(loss)

            if TBWriter:
                TBWriter.add_scalar("Loss/test/"+key, loss, epoch)

        if printData:
            missclassfied['pred']=np.concatenate(missclassfied['pred'])
            missclassfied['target']=np.concatenate(missclassfied['target'])
            missclassfied['X']=np.concatenate(missclassfied['X'])
            missclassfied['Xs']=np.concatenate(missclassfied['Xs'])

            with open(missclassfied['filename'] , 'wb') as handle:
                pickle.dump(missclassfied, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('wrong classfied test cases are saved to ',missclassfied['filename'] )
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
@click.option('--batchsize', type=int, default=512, help='batch size [512]')
@click.option('--epochs', type=int, default=500, help='training epochs [500]')
@click.option('--gpu', type=int, default=0, help='GPU id [0]')
@click.option('--resol', default=5000, help='resolution [5000]')
@click.option('-n', type=int, default=10, help='sampling n samples from database; -1 for all [10]')
@click.option('--bedpe',type=str,default=None, help = '.bedpe file containing labelling cases')
@click.option('--test', type=str, default=None, help='comma separated test files in .bcool')
@click.option('--reference', type=str, default=None, help='a file contains reference panel')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=20, help="peak window size: (2w+1)x(2w+1) [20]")
@click.option('--encoding_dim',type = int, default =64,help='encoding dim [64]')
@click.option('--feature',type = str, default = '1,2', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank  [1,2]')
@click.option('--ti',type = int, default = None, help = 'use the indexed sample from the test group during training if multiple existed; values between [0,n)')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')

@click.option('--check_point',type=str,default=None,help='checkpoint')
@click.option('--cnn',type=bool,default=True,help='cnn encoder [True]')
@click.option('--useadam',type=bool,default=True,help='USE adam [True]')
@click.option('--lm',type=bool,default=True,help='large memory')
@click.option('--cawr',type=bool,default=False,help ='CosineAnnealingWarmRestarts [False]')
@click.argument('trainingset', type=str,required=True)
@click.argument('prefix', type=str,required=True)
def train(cawr,lm,useadam,cnn,check_point,prefix,lr,batchsize, epochs, gpu, trainingset, resol, n, test, reference, bedpe, max_distance,w,feature,ti,encoding_dim,eval_ti):
    """Train RefHiC for TAD boundary annotation

    \b
    TRAININGSET: training data in .pkl
    PREFIX: output prefix
    """
    parameters = {'cnn': cnn, 'w': w, 'feature': feature, 'resol': resol, 'encoding_dim': encoding_dim,'model':'refhicNet-tad','classes':2}
    if checkConfig():
        pass
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()

    reference = referenceMeta(reference)

    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+"cuda:"+str(gpu))
    else:
        device = torch.device("cpu")
    if lm:
        occccccc = torch.zeros((256,1024,18000)).to(device)
    chromTest = {'chr15','chr16','chr17',15,16,17,'15','16','17'}
    chromVal = {'chr11','chr12',11,12,'11','12'}
    _mask = np.zeros(2 * (w * 2 + 1) ** 2 + 2 * (2 * w + 1) + 4)
    featureMask = feature.split(',')
    if '0' in featureMask:
        _mask[:] = 1
    if '1' in featureMask:
        _mask[:(2 * w + 1) ** 2] = 1
    if '2' in featureMask:
        _mask[(2 * w + 1) ** 2:2 * (2 * w + 1) ** 2] = 1
    if '3' in featureMask:
        _mask[2 * (2 * w + 1) ** 2:2 * (2 * w + 1) ** 2 + 2 * (2 * w + 1)] = 1
    if '4' in featureMask:
        _mask[2 * (2 * w + 1) ** 2 + 2 * (2 * w + 1)] = 1
    if '5' in featureMask:
        _mask[2 * (2 * w + 1) ** 2 + 2 * (2 * w + 1) + 1] = 1
    if '6' in featureMask:
        _mask[2 * (2 * w + 1) ** 2 + 2 * (2 * w + 1) + 2] = 1
    if '7' in featureMask:
        _mask[2 * (2 * w + 1) ** 2 + 2 * (2 * w + 1) + 3] = 1
    featureMask = np.ma.make_mask(_mask)
    print('#features',np.sum(featureMask))

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


    if test is not None and bedpe is not None:
        testBcools  = [bcool(file_path+'::/resolutions/'+str(resol)) for file_path in test.split(',')]
        extraBcools = [bcool(file_path+'::/resolutions/'+str(resol)) for file_path in reference['file'].to_list()]
        _bedpe = pd.read_csv(bedpe, header=None, sep='\t')

        # read labels
        labels = {}

        for _, chr1, pos1, _, chr2, pos2, _, labelleft,labelright in _bedpe.itertuples():
            label=[labelleft,labelright]
            if chr1 == chr2 and abs(pos2 - pos1) < max_distance:
                if chr1 not in labels:
                    labels[chr1] = {'contact':[],'label':[]}
                labels[chr1]['contact'].append((pos1,pos2))
                labels[chr1]['label'].append(label)

        # read data of these labels
        X = {}  # for test
        Xs = {} # for extra
        label={}


        for chrom in labels:
            if chrom not in X:
                label[chrom] = []
            for i in range(len(labels[chrom]['label'])):
                label[chrom].append(labels[chrom]['label'][i])

            for g in testBcools:
                if chrom not in X:
                    X[chrom] = [[] for _ in range(len(labels[chrom]['contact']))]
                bmatrix = g.bchr(chrom, max_distance=max_distance)
                for i in range(len(labels[chrom]['contact'])):
                    x,y = labels[chrom]['contact'][i]

                    mat,meta = bmatrix.square(x,y,w,'b')
                    X[chrom][i].append(np.concatenate((mat.flatten(), meta)))

            for g in extraBcools:
                if chrom not in Xs:
                    Xs[chrom] = [[] for _ in range(len(labels[chrom]['contact']))]
                bmatrix = g.bchr(chrom, max_distance=max_distance)
                for i in range(len(labels[chrom]['contact'])):
                    x,y = labels[chrom]['contact'][i]
                    mat,meta = bmatrix.square(x,y,w,'b')
                    Xs[chrom][i].append(np.concatenate((mat.flatten(), meta)))


        X_train = []
        Xs_train = []
        y_label_train = []
        X_test = []
        Xs_test = []
        y_label_test = []
        X_val = []
        Xs_val = []
        y_label_val = []
        for chrom in X:
            for i in range(len(X[chrom])):
                x=np.asarray(X[chrom][i])[:, featureMask]
                xs=np.asarray(Xs[chrom][i])[:, featureMask]
                if chrom in chromTest:
                    X_test.append(x)
                    Xs_test.append(xs)
                    y_label_test.append(label[chrom][i])
                elif chrom in chromVal:
                    X_val.append(x)
                    Xs_val.append(xs)
                    y_label_val.append(label[chrom][i])
                else:
                    X_train.append(x)
                    Xs_train.append(xs)
                    y_label_train.append(label[chrom][i])
        with open(trainingset, 'wb') as handle:
            pickle.dump((X_train, Xs_train, y_label_train,
                     X_test, Xs_test, y_label_test,
                     X_val, Xs_val, y_label_val), handle, protocol=pickle.HIGHEST_PROTOCOL)
        del X,Xs
    elif trainingset.endswith('.pkl'):
        print('reading pkl')
        with open(trainingset, 'rb') as handle:
            X_train, Xs_train, y_label_train, \
            X_test, Xs_test, y_label_test, \
            X_val, Xs_val, y_label_val = pickle.load(handle)

    if eval_ti is None:
        eval_ti = {}
        for _i in range(X_train[0].shape[0]):
            eval_ti['sample'+str(_i)] = _i
    print('eval_ti',eval_ti)
    print('#train:',len(y_label_train))
    print('#test:', len(y_label_test))
    print('#validation:', len(y_label_val))

    prefix = prefix+'_feature'+str(feature)


    print('#training cases',len(y_label_train))
    if n == -1:
        n = None

    training_data = inMemoryDataset(X_train,Xs_train,y_label_train,samples=n,ti=ti)


    randGen = torch.Generator()
    randGen.manual_seed(42)
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True,num_workers=1,worker_init_fn = seed_worker)


    val_dataloaders = {}
    for _k in eval_ti:
        val_data = inMemoryDataset(X_val,Xs_val,y_label_val,samples=None,ti=eval_ti[_k])
        val_dataloaders[_k] = DataLoader(val_data, batch_size=batchsize, shuffle=True,num_workers=1)



    earlyStopping = {'patience':200000000,'loss':np.inf,'wait':0,'model_state_dict':None,'epoch':0,'parameters':None}

    model = refhicNet(np.sum(featureMask),encoding_dim=encoding_dim,CNNencoder=cnn,win=2*w+1,classes=2).to(device)
    ema = EMA(model.parameters(), decay=0.999)

    lossfn = torch.nn.MSELoss()

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
        scheduler2 = CosineScheduler(int(epochs*0.95), warmup_steps=0, base_lr=lr, final_lr=1e-6)


    for epoch in tqdm(range(0, epochs)):


        if not cawr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler2(epoch)

        trainLoss=trainModel(model,train_dataloader, optimizer, lossfn, epoch, device,batchsize,TBWriter=TBWriter,scheduler=scheduler,ema=ema)
        testLoss=testModel(model,val_dataloaders, lossfn,device,epoch,TBWriter=TBWriter,printData= False,ema=None)


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
