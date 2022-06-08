import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5py
from groupLoopModels import attentionToAdditionalHiC,baseline,focalLoss,elrBCE_loss
import numpy as np
from gcooler import gcool
import click
from data import  inMemoryDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import datetime
import pickle
from coteachingPlus import gen_forget_rate,coteachingPlusTrain

def trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize=128,TBWriter=None,baseline=False,elr=False):
    model.train()
    preds=[]
    targets = []
    indices = []
    for batch_idx,X in enumerate(train_dataloader):

        target = X[-1].to(device)
        optimizer.zero_grad()
        if not baseline:
            output = model(X[1].to(device),X[2].to(device))
        else:
            output = model(X[1].to(device))

        preds.append(output.cpu())
        targets.append(target.cpu())
        if elr:
            # print('elr loss')
            loss = criterion(X[0],output.view(-1), target.view(-1))
            indices.append(X[0].cpu())
        else:
            # print('bce loss')
            loss = criterion(output, target.view(-1, 1))

        loss.backward()
        optimizer.step()


    preds=torch.vstack(preds).flatten()
    targets=torch.vstack(targets).flatten()
    if elr:
        indices = torch.hstack(indices)
        loss = criterion(indices.cuda().view(-1),preds.cuda().view(-1), targets.cuda().view(-1),train=False)
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
    return loss



def testModel(model, test_dataloaders,criterion, device,epoch,printData=False,TBWriter=None,baseline=False,elr=False):
    model.eval()
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
                target = X[-1].to(device)


                if not baseline:
                    output = model(X[1].to(device), X[2].to(device))
                else:
                    output = model(X[1].to(device))
                if printData:
                    failed= ((torch.sigmoid(output)>0.5)*1!=target).cpu().numpy().flatten()
                    print('#failed',np.sum(failed))
                    missclassfied['pred'].append(torch.sigmoid(output).cpu().numpy().flatten()[failed])
                    missclassfied['target'].append(target.cpu().numpy().flatten()[failed])
                    missclassfied['X'].append(X[1][failed,...].numpy())
                    missclassfied['Xs'].append(X[2][failed,...].numpy())

                preds.append(output.cpu())
                targets.append(target.cpu())
                indices.append(X[0].cpu())





            preds=torch.vstack(preds).flatten()
            targets=torch.vstack(targets).flatten()
            if elr:
                indices = torch.hstack(indices).flatten()
                loss = torch.mean(criterion(indices.cuda().view(-1), preds.cuda().view(-1), targets.cuda().view(-1),train=False))
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
@click.option('--epochs', type=int, default=20, help='training epochs')
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
@click.option('--encoding_dim',type = int, default =128,help='encoding dim')
@click.option('--oversampling',type=float, default = 1.0, help ='oversampling positive training cases, [0-2]')
@click.option('--feature',type = str, default = 0, help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--ti',type = int, default = 0, help = 'use the indexed sample from the test group during training if multiple existed; values between [0,n)')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')
@click.option('--models',type=str,default ='groupLoop',help='groupLoop; baseline; groupLoop,baseline')
@click.option('--coteach',type=bool,default=True,help = 'co-teach training')
@click.option('--fr',type=float,default=0,help='forget rate')
@click.option('--elr',type=bool,default=False, help ='elr loss training')
@click.option('--elr_lambda',type=float,default=3, help ='elr Lambda; need to be very careful')
@click.option('--elr_beta',type=float,default=0.9, help ='elr beta')
@click.option('--prefix',type=str,default='',help='output prefix')
def trainAttention(prefix,lr,fr,elr,elr_lambda,elr_beta,name,batchsize, epochs, gpu, trainingset, skipchrom, resol, n, test, extra, bedpe, max_distance,w,oversampling,feature,ti,encoding_dim,models,eval_ti,coteach):
    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+"cuda:"+str(gpu))
    else:
        device = torch.device("cpu")

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
            _oversamples = np.random.choice(training_index[1], size=int(len(training_index[1])*(oversampling-1)), replace=False)
            training_index[1] = training_index[1] + list(_oversamples)
        else:
            _samples = np.random.choice(training_index[1], size=int(len(training_index[1]) * oversampling),
                                               replace=False)
            training_index[1] = list(_samples)
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

    for i in range(len(X_train)):
        X_train[i]= X_train[i][ti, :][None]
        Xs_train[i] = np.repeat(X_train[i],30,axis=0)
    for i in range(len(X_test)):
        X_test[i]= X_test[i][ti, :][None]
        Xs_test[i] = np.repeat(X_test[i],30,axis=0)
    for i in range(len(X_val)):
        X_val[i]= X_val[i][ti, :][None]
        Xs_val[i] = np.repeat(X_val[i],30,axis=0)

    ti=0
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
        if coteach:
            print('co-teaching')

            model = attentionToAdditionalHiC(np.sum(featureMask),encoding_dim=encoding_dim).to(device)
            model2 = attentionToAdditionalHiC(np.sum(featureMask), encoding_dim=encoding_dim).to(device)
            TBWriter = SummaryWriter(comment=' '+name)
            model.train()
            model2.train()
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,eps=1e-8)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, eps=1e-8)

            fr_schedule = gen_forget_rate(epochs,40,fr)
            print('fr_schedule',fr_schedule)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,180], gamma=0.1)
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[100, 180], gamma=0.1)
            for epoch in tqdm(range(0, epochs)):
                coteachingPlusTrain(model,model2,optimizer,optimizer2,train_dataloader,criterion,epoch,fr_schedule,device,batchsize,TBWriter,init_epoch=5)
                testModel(model,val_dataloaders, criterion,device,epoch,TBWriter=TBWriter)
                # scheduler.step()
                # scheduler2.step()

        else:
            print('without co-teaching')
            earlyStopping = {'patience':5,'loss':np.inf,'wait':0,'state':None,'epoch':0}
            model = attentionToAdditionalHiC(np.sum(featureMask),encoding_dim=encoding_dim).to(device)
            TBWriter = SummaryWriter(comment=' '+name)
            model.train()
            if elr:
                print('............',len(y_label_train),'....................')
                criterion = elrBCE_loss(len(y_label_train),device=device,Lambda=elr_lambda,beta=elr_beta)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,eps=1e-8)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,180], gamma=0.1)
            for epoch in tqdm(range(0, epochs)):
                if elr:
                    trainModel(model, train_dataloader, optimizer, criterion, epoch, device, batchsize,
                               TBWriter=TBWriter, elr=True)
                    testModel(model, val_dataloaders, criterion, device, epoch, TBWriter=TBWriter,elr=True,printData= (epoch == epochs-1))
                else:
                    trainLoss=trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize,TBWriter=TBWriter,elr=False)
                    testLoss=testModel(model,val_dataloaders, criterion,device,epoch,TBWriter=TBWriter,elr=False,printData= (epoch == epochs-1))

                if testLoss < earlyStopping['loss'] and earlyStopping['wait']<earlyStopping['patience']:
                    earlyStopping['loss'] = testLoss
                    earlyStopping['epoch'] = epoch
                    earlyStopping['wait'] = 0
                    earlyStopping['state'] = model.state_dict()
                else:
                    earlyStopping['wait'] += 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': testLoss,
                }, prefix+'_groupLoop_epoch'+str(epoch)+'.tar')

        print('finsh trying; best model with early stopping is epoch: ',earlyStopping['epoch'], 'loss is ',earlyStopping['loss'])
        torch.save(earlyStopping['state'], prefix+'_groupLoop_bestModel_state.pt')



    if 'baseline' in models.lower():
        # baseline model
        baselineModel = baseline(np.sum(featureMask),encoding_dim=encoding_dim).to(device)
        TBWriterB = SummaryWriter(comment=' baseline '+name)
        baselineModel.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizerB = torch.optim.Adam(baselineModel.parameters(), lr=lr,eps=1e-8)
        schedulerB = torch.optim.lr_scheduler.MultiStepLR(optimizerB, milestones=[40,80,180], gamma=0.5)

        earlyStopping = {'patience': 5, 'loss': np.inf, 'wait': 0, 'state': None, 'epoch': 0}

        for epoch in tqdm(range(epochs)):
            trainModel(baselineModel,train_dataloader, optimizerB, criterion, epoch, device,batchsize,TBWriter=TBWriterB,baseline=True)
            testLoss=testModel(baselineModel,val_dataloaders, criterion,device,epoch,TBWriter=TBWriterB,baseline=True)
            # schedulerB.step()
            if testLoss < earlyStopping['loss'] and earlyStopping['wait'] < earlyStopping['patience']:
                earlyStopping['loss'] = testLoss
                earlyStopping['epoch'] = epoch
                earlyStopping['wait'] = 0
                earlyStopping['state'] = baselineModel.state_dict()
            else:
                earlyStopping['wait'] += 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': baselineModel.state_dict(),
                'optimizer_state_dict': optimizerB.state_dict(),
                'loss': testLoss,
            }, prefix + '_baseline_epoch' + str(epoch) + '.tar')

        print('finsh trying; best model with early stopping is epoch: ',earlyStopping['epoch'])
        torch.save(earlyStopping['state'], prefix+'_baseline_bestModel_state.pt')
    #
    # return model






if __name__ == '__main__':
    trainAttention()