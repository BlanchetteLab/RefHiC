import torch
import pandas as pd
from torch.utils.data import DataLoader
import h5py
from torchlars import LARS
from groupLoopModels import attentionToAdditionalHiC,baseline,focalLoss,distilFocalLoss#, EMA
from torch_ema import ExponentialMovingAverage as EMA
from groupLoopModels import baseline as teacherModel
import numpy as np
from gcooler import gcool
import click
from data import  inMemoryDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import datetime
import pickle
from pretrain import contrastivePretrain
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








def trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize=128,TBWriter=None,baseline=False,scheduler=None,teacher=None,ema=None):
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



def testModel(model, test_dataloaders,criterion, device,epoch,printData=False,TBWriter=None,baseline=False,teacher=None,ema=None):
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
@click.option('--lr', type=float, default=1e-3, help='learning rate')
@click.option('--name',type=str,default='', help ='training name')
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--epochs', type=int, default=30, help='training epochs')
@click.option('--gpu', type=int, default=0, help='GPU training')
@click.option('--trainingset', type=str, required=True, help='training data in .pkl or .h5 file; use if file existed; otherwise, prepare one for reuse')
@click.option('--skipchrom', type=str, default=None, help='skip one chromosome for during training')
@click.option('--resol', default=5000, help='resolution')
@click.option('-n', type=int, default=10, help='sampling n samples from database; -1 for all')
@click.option('--bedpe',type=str,default=None, help = '.bedpe file containing labelling cases')
@click.option('--test', type=str, default=None, help='comma separated test files in .gcool')
@click.option('--extra', type=str, default=None, help='a file contain a list of extra .gcools (i.e. database)')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1)")
@click.option('--encoding_dim',type = int, default =64,help='encoding dim')
@click.option('--oversampling',type=float, default = 1.0, help ='oversampling positive training cases, [0-2]')
@click.option('--feature',type = str, default = '1,2,3,4,5', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--ti',type = int, default = None, help = 'use the indexed sample from the test group during training if multiple existed; values between [0,n)')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')
@click.option('--models',type=str,default ='groupLoop',help='groupLoop; baseline; groupLoop,baseline')
@click.option('--prefix',type=str,default='',help='output prefix')
@click.option('--teacher',type=str,default=None,help='teacher model')
@click.option('--check_point',type=str,default=None,help='checkpoint')
@click.option('--pw',type=float,default=-1,help='alpha for focal loss')
@click.option('--cnn',type=bool,default=True,help='cnn encoder')
@click.option('--useadam',type=bool,default=False,help='USE adam')
@click.option('--lm',type=bool,default=True,help='large memory')
@click.option('--useallneg',type=bool,default=False,help='use all negative cases')
@click.option('--cawr',type=bool,default=False,help ='CosineAnnealingWarmRestarts')
def trainAttention(useallneg,cawr,lm,useadam,cnn,pw,check_point,teacher,prefix,lr,name,batchsize, epochs, gpu, trainingset, skipchrom, resol, n, test, extra, bedpe, max_distance,w,oversampling,feature,ti,encoding_dim,models,eval_ti):
    # if feature =='1,2':
    #     lr=0.001
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


    if test is not None and extra is not None and bedpe is not None:
        testGcools  = [gcool(file_path+'::/resolutions/'+str(resol)) for file_path in test.split(',')]
        extraGcools = [gcool(file_path+'::/resolutions/'+str(resol)) for file_path in pd.read_csv(extra, header=None)[0].to_list()]
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

            for g in testGcools:
                if chrom not in X:
                    X[chrom] = [[] for _ in range(len(labels[chrom]['contact']))]
                bmatrix = g.bchr(chrom, max_distance=max_distance)
                for i in range(len(labels[chrom]['contact'])):
                    x,y = labels[chrom]['contact'][i]

                    mat,meta = bmatrix.square(x,y,w,'b')
                    X[chrom][i].append(np.concatenate((mat.flatten(), meta)))

            for g in extraGcools:
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
        with open(trainingset + '.pkl', 'wb') as handle:
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

    if skipchrom is None:
        prefix = prefix+'_feature'+str(feature)
    else:
        prefix = prefix+'_feature'+str(feature) + '_chr' + str(skipchrom)


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


    if 'grouploop' in models.lower():
        print('without co-teaching')
        earlyStopping = {'patience':200000000,'loss':np.inf,'wait':0,'state':None,'epoch':0}

        model = attentionToAdditionalHiC(np.sum(featureMask),encoding_dim=encoding_dim,CNNencoder=cnn,win=2*w+1,classes=2).to(device)
        ema = EMA(model.parameters(), decay=0.999)

        lossfn = torch.nn.MSELoss()

        TBWriter = SummaryWriter(comment=' '+name)

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

            trainLoss=trainModel(model,train_dataloader, optimizer, lossfn, epoch, device,batchsize,TBWriter=TBWriter,scheduler=scheduler,teacher=teacher,ema=ema)
            testLoss=testModel(model,val_dataloaders, lossfn,device,epoch,TBWriter=TBWriter,printData= False,teacher=teacher,ema=None)


            if testLoss < earlyStopping['loss'] and earlyStopping['wait']<earlyStopping['patience']:
                earlyStopping['loss'] = testLoss
                earlyStopping['epoch'] = epoch
                earlyStopping['wait'] = 0
                earlyStopping['state'] = model.state_dict()
            else:
                earlyStopping['wait'] += 1
            if epoch%10==0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': testLoss,
                }, prefix+'_groupLoop_epoch'+str(epoch)+'.tar')

                if ema:
                    ema.store()
                    ema.copy_to()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': testLoss,
                    }, prefix+'_groupLoop_epoch'+str(epoch)+'_ema.tar')
                    ema.restore()

        print('finsh trying; best model with early stopping is epoch: ',earlyStopping['epoch'], 'loss is ',earlyStopping['loss'])
        torch.save(earlyStopping['state'], prefix+'_groupLoop_bestModel_state.pt')








if __name__ == '__main__':
    trainAttention()
