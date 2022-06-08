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



def hardNegAug(X,Xs, y):
    # torch.Size([512]) torch.Size([512, 926]) torch.Size([512, 10, 926])
    # print(X.shape,Xs.shape,y.shape)
    positiveXs = Xs[(y == 1).flatten(), :, :]
    positiveX = X[(y == 1).flatten()]
    negativeXs = Xs[(y == 0).flatten(), :, :]
    negativeX = X[(y == 0).flatten()]
    topk=X.shape[0]//10

    _n = np.max([negativeX.shape[0]//20,5])
    n = negativeX.shape[0]
    _negativeX = negativeX[np.random.choice(negativeX.shape[0], _n, replace=True)]
    _fakeNegativeXs = positiveXs[np.random.choice(positiveXs.shape[0], _n, replace=True),...]
    # print(negativeXs.shape,negativeX.shape,' -->')
    negativeX=torch.cat([negativeX,_negativeX],dim=0)
    negativeXs = torch.cat([negativeXs,_fakeNegativeXs],dim=0)
    negativeX = negativeX[np.random.choice(negativeX.shape[0], n, replace=False)]
    negativeXs = negativeXs[np.random.choice(negativeXs.shape[0], n, replace=False), ...]

    X = torch.cat([positiveX,negativeX],dim=0).to(X.device)
    Xs = torch.cat([positiveXs, negativeXs], dim=0).to(Xs.device)
    y = torch.cat([torch.ones(positiveX.shape[0],1),torch.zeros(negativeX.shape[0],1)]).to(y.device)
    idx = torch.randperm(y.shape[0])[:topk].to(y.device)
    return X[idx,...],Xs[idx,...],y[idx,...]





def trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize=128,TBWriter=None,baseline=False,scheduler=None,teacher=None,ema=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    preds=[]
    teacherLogits=[]
    targets = []

    iters = len(train_dataloader)

    for batch_idx,X in enumerate(train_dataloader):
        target = X[-1].to(device)
        X[1] = X[1].to(device)

        # skip=(target.flatten() == 1) & (X[1][:, -1]<0.1)
        # X[1] = X[1][skip==False,...]
        # X[2] = X[2][skip==False,...]
        # target = target[skip==False,...]

        X[2] = X[2].to(device)

        '''
        # # hard neg
        _x, _xs, _target=hardNegAug(X[1], X[2], target)
        # # print(X[1].shape, X[2].shape, target.shape,' -->')
        X[1] = torch.cat((X[1],_x),0)
        X[2] = torch.cat((X[2], _xs), 0)
        target = torch.cat((target, _target), 0)
        # # . hard neg
        '''

        optimizer.zero_grad()
        if not baseline:
            output = model(X[1],X[2])
            if isinstance(criterion,distilFocalLoss):
                teacherLogit = teacher(X[1])
        else:
            output = model(X[1].to(device))

        preds.append(torch.sigmoid(output[...,[-1]].cpu()))
        targets.append(target.cpu())
        if isinstance(criterion, distilFocalLoss):
            teacherLogits.append(teacherLogit.cpu())
            loss = criterion(output, target,teacherLogit)
        else:
            # print(output.shape,target.shape,'.....')
            loss = criterion(output, target.repeat(1,output.shape[-1]))


        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if ema:
            ema.update()
        if scheduler:
            scheduler.step(epoch + batch_idx/ iters)
    # print('extralosses',extralosses)


    preds=torch.vstack(preds).flatten()
    targets=torch.vstack(targets).flatten()
    if isinstance(criterion,distilFocalLoss):
        teacherLogits = torch.vstack(teacherLogits).flatten()
        loss = criterion(preds,targets,teacherLogits)
    else:
        loss = criterion(preds, targets)
    acc = torchmetrics.functional.accuracy(preds, targets.to(int))
    f1 = torchmetrics.functional.f1(preds, targets.to(int))
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
    # import sys
    # sys.exit(0)
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
            teacherLogits=[]
            for X in test_dataloader:
                target = X[-1]

                # skip = (target.flatten() == 1) & (X[1][:, -1] < 0.1)
                # X[1] = X[1][skip == False, ...]
                # X[2] = X[2][skip == False, ...]
                # target = target[skip == False, ...]


                target=target.to(device)


                if not baseline:
                    X[1]=X[1].to(device)
                    X[2]=X[2].to(device)
                    output = model(X[1],X[2])
                    if isinstance(criterion, distilFocalLoss):
                        teacherLogit = teacher(X[1])
                else:
                    output = model(X[1].to(device))
                output=torch.sigmoid(output)
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
                if isinstance(criterion, distilFocalLoss):
                    teacherLogits.append(teacherLogit.cpu())

            preds=torch.vstack(preds).flatten()
            targets=torch.vstack(targets).flatten()
            if isinstance(criterion, distilFocalLoss):
                teacherLogits = torch.vstack(teacherLogits).flatten()

                loss = torch.mean(criterion(preds, targets,teacherLogits))
            else:
                loss = torch.mean(criterion(preds, targets))
            losses.append(loss)
            acc = torchmetrics.functional.accuracy(preds, targets.to(int))
            f1 = torchmetrics.functional.f1(preds, targets.to(int))
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
        for _, chr1, pos1, _, chr2, pos2, _, label in _bedpe.itertuples():
            if chr1 == chr2 and abs(pos2 - pos1) < max_distance:
                if chr1 not in labels:
                    labels[chr1] = {'contact':[],'label':[]}
                labels[chr1]['contact'].append((pos1,pos2))
                labels[chr1]['label'].append(label)

        # read data of these labels
        X = {}  # for test
        Xs = {} # for extra
        trainingset_H5 = h5py.File(trainingset, 'w')
        trainingset_H5.create_group('feature')
        trainingset_H5.create_group('label')
        idx = 0
        for chrom in labels:
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

            # write data to hdf5
            for i in range(len(labels[chrom]['label'])):
                _Xs = np.vstack(Xs[chrom][i])
                _X  = np.vstack(X[chrom][i])
                _target = labels[chrom]['label'][i]
                _bp1, _bp2 = labels[chrom]['contact'][i]
                trainingset_H5['label'].create_dataset(str(idx),data=np.asarray([idx, int(chrom.replace('chr', '')), _bp1, _bp2, _target]))
                grp=trainingset_H5['feature'].create_group(str(idx))
                grp.create_dataset('X', data=_X)
                grp.create_dataset('Xs', data=_Xs)
                idx = idx+1
            del Xs[chrom],X[chrom]
        trainingset_H5.attrs['num'] = idx
        trainingset_H5.close()

    print("go ")

    # load labels with index from h5
    if skipchrom is not None:
        _skipchrom = int(skipchrom.replace('chr',''))
    else:
        _skipchrom = None
    if trainingset.endswith('.h5') or trainingset.endswith('.hdf5'):
        print('reading hdf5')
        trainingset_H5 = h5py.File(trainingset, 'r')
        training_index = {0:[],1:[]}


        for i in range(trainingset_H5.attrs['num']):
            value = trainingset_H5['label/'+str(i)][()]
            if value[1] != _skipchrom:
                training_index[value[-1]].append(str(value[0])+'_'+str(value[1]))


        if oversampling>1:
            _oversamples = np.random.choice(training_index[1], size=int(len(training_index[1])*(oversampling-1)), replace=True)
            training_index[1] = training_index[1] + list(_oversamples)
        else:
            _samples = np.random.choice(training_index[1], size=int(len(training_index[1]) * oversampling),
                                               replace=False)
            training_index[1] = list(_samples)
        if useallneg:
            pass
        else:
            training_index[0] = np.random.choice(training_index[0], size=int(len(training_index[1])),replace=False)

        y_label = []
        y_idx = []
        for key in training_index:
            y_label = y_label + [key]*len(training_index[key])
            y_idx = y_idx + list(training_index[key])
        y_label = np.asarray(y_label)
        y_idx = np.asarray(y_idx)
        _argsort_idx = np.argsort(y_idx)
        y_label = y_label[_argsort_idx]
        y_idx = y_idx[_argsort_idx]

        print(len(y_label),len(y_idx),y_label.sum())
        X_train = []
        Xs_train = []
        y_label_train = []
        X_test = []
        Xs_test = []
        y_label_test = []
        X_val = []
        Xs_val = []
        y_label_val = []
        grp = trainingset_H5['feature']
        print('load data')
        for i in range(len(y_idx)):
            idx, chrom = y_idx[i].split('_')
            x = grp[idx]['X'][:, featureMask]
            xs = grp[idx]['Xs'][:, featureMask]
            if chrom in chromTest:
                X_test.append(x)
                Xs_test.append(xs)
                y_label_test.append(y_label[i])
            elif chrom in chromVal:
                X_val.append(x)
                Xs_val.append(xs)
                y_label_val.append(y_label[i])
            else:
                X_train.append(x)
                Xs_train.append(xs)
                y_label_train.append(y_label[i])
        with open(trainingset+'.pkl', 'wb') as handle:
            pickle.dump((X_train,Xs_train,y_label_train,
                         X_test,Xs_test,y_label_test,
                         X_val,Xs_val,y_label_val),handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif trainingset.endswith('.pkl'):
        print('reading pkl')
        with open(trainingset, 'rb') as handle:
            X_train,Xs_train,y_label_train,\
                X_test,Xs_test,y_label_test,\
                X_val,Xs_val,y_label_val = pickle.load(handle)
            for i in range(len(X_train)):
                X_train[i] = X_train[i][:,featureMask]
                Xs_train[i] = Xs_train[i][:,featureMask]
                # Xs_train[i]=np.vstack([Xs_train[i],np.random.permutation(Xs_train[i].reshape(-1)).reshape(Xs_train[i].shape)])
            for i in range(len(X_test)):
                X_test[i] = X_test[i][:,featureMask]
                Xs_test[i] = Xs_test[i][:,featureMask]
            for i in range(len(X_val)):
                X_val[i] = X_val[i][:,featureMask]
                Xs_val[i] = Xs_val[i][:,featureMask]
                # Xs_val[i] = np.vstack(
                #     [Xs_val[i], np.random.permutation(Xs_val[i].reshape(-1)).reshape(Xs_val[i].shape)])
            # _posIdx =np.argwhere(np.asarray(y_label_train)==1).flatten()
            # for i in _posIdx:
            #     _x = X_train[i].copy()
            #     np.random.shuffle(_x.transpose()[:-1])
            #     # _x=np.random.permutation(X_train[i].reshape(-1)).reshape(X_train[i].shape)
            #     X_train.append(_x)
            #     Xs_train.append(Xs_train[i])
            #     y_label_train.append(0)
            #     X_train.append(X_train[i])
            #     Xs_train.append(Xs_train[i])
            #     y_label_train.append(1)

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


    ######################################################################################################
    ##   control set
    # with open('20K_zeroLoop.h5.pkl', 'rb') as handle:
    #     _, _, _, _, _, _, \
    #     _X_val, _Xs_val, _y_label_val = pickle.load(handle)
    #
    #     for i in range(len(_X_val)):
    #         _X_val[i] = _X_val[i][:, featureMask]
    #         _Xs_val[i] = _Xs_val[i][:, featureMask]
    #         _y_label_val[i] = 0
    # val_data = inMemoryDataset(_X_val, _Xs_val, _y_label_val, samples=None, ti=0)
    # val_dataloaders['control'] = DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=1)
    #
    # with open('gm12878_chr3Aschr2.h5.pkl', 'rb') as handle:
    #     _X_val, _Xs_val, _y_label_val,_, _, _, _, _, _ = pickle.load(handle)
    #
    #     for i in range(len(_X_val)):
    #         _X_val[i] = _X_val[i][:, featureMask]
    #         _Xs_val[i] = _Xs_val[i][:, featureMask]
    # val_data = inMemoryDataset(_X_val, _Xs_val, _y_label_val, samples=None, ti=0)
    # val_dataloaders['gm12878_chr3Aschr2'] = DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=1)


    ######################################################################################################


    if 'grouploop' in models.lower():
        print('without co-teaching')
        earlyStopping = {'patience':200000000,'loss':np.inf,'wait':0,'state':None,'epoch':0}

        model = attentionToAdditionalHiC(np.sum(featureMask),encoding_dim=encoding_dim,CNNencoder=cnn,win=2*w+1).to(device)
        ema = EMA(model.parameters(), decay=0.999)

        # ema =None

        if teacher is not None:
            _param = torch.load(teacher, map_location='cuda:'+str(gpu))
            teacher = teacherModel(np.sum(featureMask),encoding_dim=64).to(device)
            teacher.load_state_dict(_param,strict=False)
            criterion = distilFocalLoss(alpha=pw, gamma=2,beta=0.25)
            # criterion = focalLoss(alpha=-1, gamma=2, adaptive=False)

        else:
            criterion = focalLoss(alpha=pw, gamma=2, adaptive=True)
            #criterion = torch.nn.BCEWithLogitsLoss()

        ### pretrain
        if check_point is None:
            model=contrastivePretrain(model,pretrain_dataloader,0.001,20,device)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, prefix + '_groupLoop_pretrain.tar')
            pass
        ### end of pretrain

        # for name, param in model.named_parameters():
        #     print(name,"\n",param,"\n------------------")
        # import sys
        # sys.exit(0)

        TBWriter = SummaryWriter(comment=' '+name)

        model.train()

        if check_point:
            _modelstate = torch.load(check_point, map_location='cuda:' + str(gpu))
            if 'model_state_dict' in _modelstate:
                _modelstate = _modelstate['model_state_dict']
            model.load_state_dict(_modelstate)




        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True)
        # optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)
        if useadam:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.1)

        if cawr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
        else:
            scheduler = None
            scheduler2 = CosineScheduler(int(epochs*0.95), warmup_steps=0, base_lr=lr, final_lr=1e-6)


        for epoch in tqdm(range(0, epochs)):
            # if epoch==6:
            #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
            #     scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
            #     schedulerLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200000, 5000000], gamma=0.1)
            #     scheduler2 = CosineScheduler(1000, warmup_steps=5, base_lr=lr, final_lr=lr)
            # if epoch==5:
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)

            if not cawr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler2(epoch)
            # if epoch>20:
            #     # pass
            #     criterion = distilFocalLoss(alpha=-1, gamma=2, beta=10)

            # scheduler2.step()
            # if epoch==20:
            #     optimizer = torch.optim.SGD(
            #         [{'params': list(model.singledataFC.parameters()) + list(model.attentionFC.parameters()) + list(
            #                 model.predictor.parameters()), 'lr': lr}
            #         ], lr=lr, momentum=0.9, nesterov=True)


            # if epoch > 50:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = scheduler2(epoch-50)

            # if epoch==50:
            #     for name, param in model.named_parameters():
            #         if 'encoder' in name or 'att' == name:
            #             param.requires_grad = True
            #     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
            #                                   eps=1e-8, amsgrad=True)
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
            if epoch==50:
                pass
                # for name,param in model.named_parameters():
                #     if 'encoder' in name or 'att' == name:
                #         param.requires_grad=False
                # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                #                              eps=1e-8,amsgrad=True)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)
                # criterion = focalLoss(alpha=-1, gamma=3)

                # if teacher is None:
                #     criterion = focalLoss(alpha=0.1, gamma=2)
                # else:
                #     criterion = distilFocalLoss(alpha=-1,gamma=2,beta=2)


            trainLoss=trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize,TBWriter=TBWriter,scheduler=scheduler,teacher=teacher,ema=ema)
            testLoss=testModel(model,val_dataloaders, criterion,device,epoch,TBWriter=TBWriter,printData= (epoch == epochs-1),teacher=teacher,ema=None)

            # schedulerLR.step()


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


    if 'baseline' in models.lower():
        # baseline model
        baselineModel = baseline(np.sum(featureMask),encoding_dim=encoding_dim).to(device)
        ema = EMA(baselineModel.parameters(), decay=0.999)
        TBWriterB = SummaryWriter(comment=' baseline '+name)
        baselineModel.train()
        #criterion = torch.nn.BCEWithLogitsLoss()
        criterion = focalLoss(alpha=pw, gamma=2,adaptive=True)
        optimizer = torch.optim.Adam(baselineModel.parameters(), lr=lr, eps=1e-8, amsgrad=True)
        scheduler = None#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
        scheduler2 = CosineScheduler(int(epochs * 0.95), warmup_steps=0, base_lr=lr, final_lr=1e-6)

        earlyStopping = {'patience': 500000, 'loss': np.inf, 'wait': 0, 'state': None, 'epoch': 0}

        for epoch in tqdm(range(epochs)):
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler2(epoch)
            trainModel(baselineModel,train_dataloader, optimizer, criterion, epoch, device,batchsize,TBWriter=TBWriterB,baseline=True,scheduler=scheduler,ema=ema)
            testLoss=testModel(baselineModel,val_dataloaders, criterion,device,epoch,TBWriter=TBWriterB,baseline=True)

            if testLoss < earlyStopping['loss'] and earlyStopping['wait'] < earlyStopping['patience']:
                earlyStopping['loss'] = testLoss
                earlyStopping['epoch'] = epoch
                earlyStopping['wait'] = 0
                earlyStopping['state'] = baselineModel.state_dict()
            else:
                earlyStopping['wait'] += 1
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': baselineModel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': testLoss,
                }, prefix + '_baseline_epoch' + str(epoch) + '.tar')
                if ema:
                    ema.store()
                    ema.copy_to()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': baselineModel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': testLoss,
                    }, prefix + '_baseline_epoch' + str(epoch) + '_ema.tar')
                    ema.restore()


        print('finsh trying; best model with early stopping is epoch: ', earlyStopping['epoch'], 'loss is ',
              earlyStopping['loss'])
        torch.save(earlyStopping['state'], prefix+'_baseline_bestModel_state.pt')
    #
    # return model






if __name__ == '__main__':
    trainAttention()
