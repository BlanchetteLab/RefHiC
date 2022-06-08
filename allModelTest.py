import torch
import pandas as pd
from torch.utils.data import DataLoader
import h5py
from sklearn.ensemble import RandomForestClassifier as forest
from groupLoopModels import attentionToAdditionalHiC,baseline
import numpy as np
from gcooler import gcool
import click
from data import  inMemoryDataset
from tqdm import tqdm
import torchmetrics
import pickle
from matplotlib import pylab as plt
import joblib
from matplotlib_venn import venn3,venn3_circles
import random
from scipy import stats
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def buildmatrix(windows):
    """
    Generate training set
    :param coords: List of tuples containing coord bins
    :param width: Distance added to center. width=5 makes 11x11 windows
    :return: yields paired positive/negative samples for training
    """


    _windows = []
    for window in windows:
        window = window.reshape(21,21)
        # print(window)
        ls = window.shape[0]
        width = 10
        center = window[width, width]
        p2LL = center/(np.mean(window[ls-1-ls//4:ls, :1+ls//4])+1e-5)
        indicatar_vars = np.array([p2LL])
        ranks = stats.rankdata(window, method='ordinal')
        window = np.hstack((window.flatten(), ranks, indicatar_vars))
        window = window.flatten()
        if window.size == 1 + 2 * (2 * width + 1) ** 2 and np.all(np.isfinite(window)):
            _windows.append(window)

    windows = np.vstack(_windows)
    # print(np.max(windows))
    return windows
@click.command()


@click.option('--batchsize', type=int, default=512, help='batch size')
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
@click.option('--feature',type = str, default = '1,2,3,4,5', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')

@click.option('--grouploop',type=str,default=None,required=True, help='trained model')
@click.option('--rf',type=str,default=None,required=True, help='trained model')
@click.option('--peakachu',type=str,default=None,required=True, help='trained model')
@click.option('--mlp',type=str,default=None,required=True, help='trained model')
def testAll(batchsize, gpu, trainingset, skipchrom, resol, n, test, extra, bedpe, max_distance,w,feature,encoding_dim,eval_ti,grouploop,rf,peakachu,mlp):
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

    if eval_ti is not None:
        _eval_ti = {}
        for _ti in eval_ti.split(','):
            _i,_cov = _ti.split(':')
            _eval_ti[_cov] = int(_i)
        eval_ti = _eval_ti
    else:
        eval_ti = {'default':0}

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

    if eval_ti is None:
        eval_ti = {}
        for _i in range(X_train[0].shape[0]):
            eval_ti['sample'+str(_i)] = _i
    print('eval_ti',eval_ti)
    print('#train:',len(y_label_train))
    print('#test:', len(y_label_test))

    print('#pos/neg:',np.sum(y_label_test),len(y_label_test)-np.sum(y_label_test))



    print('#training cases',len(y_label_train))
    if n == -1:
        n = None


    test_dataloaders = {}
    for _k in eval_ti:
        test_data = inMemoryDataset(X_test,Xs_test,y_label_test,samples=None,ti=eval_ti[_k])
        test_dataloaders[_k] = DataLoader(test_data, batch_size=batchsize, shuffle=False,num_workers=1)

    grouploop_clf = attentionToAdditionalHiC(np.sum(featureMask), encoding_dim=encoding_dim).to(device)
    grouploop_clf.load_state_dict(torch.load(grouploop, map_location='cuda:'+str(gpu)))
    grouploop_clf.eval()
    mlp_clf = baseline(np.sum(featureMask),encoding_dim=encoding_dim).to(device)
    mlp_clf.load_state_dict(torch.load(mlp, map_location='cuda:'+str(gpu)))
    mlp_clf.eval()
    rf_clf = joblib.load(rf)
    peakachu_clf = joblib.load(peakachu)


    with torch.no_grad():
        for key in test_dataloaders:
            targets = []
            grouploopPreds = []
            mlpPreds = []
            rfPreds = []
            peakachuPreds = []
            cmaps = []

            for X in tqdm(test_dataloaders[key]):
                target = X[-1].to(device)
                targets.append(target.cpu())
                output = torch.sigmoid(grouploop_clf(X[1].to(device), X[2].to(device)))
                grouploopPreds.append(output.cpu())
                output = torch.sigmoid(mlp_clf(X[1].to(device)))
                mlpPreds.append(output.cpu())
                output = rf_clf.predict_proba(X[1].cpu().numpy())[:,1]
                rfPreds.append(output)
                cmaps.append(X[1][:, :21 * 21].reshape(-1,21,21))
                peakachu_input = buildmatrix(X[1][:, :21 * 21].cpu().numpy())
                # print(peakachu_input.shape)
                output = peakachu_clf.predict_proba(peakachu_input)[:,1]
                peakachuPreds.append(output)

            grouploopPreds = torch.vstack(grouploopPreds).flatten()
            cmaps = np.vstack(cmaps)
            mlpPreds = torch.vstack(mlpPreds).flatten()
            rfPreds = np.hstack(rfPreds).flatten()
            rfPreds = torch.from_numpy(rfPreds)
            peakachuPreds = np.hstack(peakachuPreds).flatten()
            peakachuPreds = torch.from_numpy(peakachuPreds)
            targets = torch.vstack(targets).flatten()
            mlpCalls = np.argwhere(mlpPreds.cpu().numpy()>0.5).flatten()
            rfCalls = np.argwhere(rfPreds.cpu().numpy()>0.5).flatten()
            peakachuCalls = np.argwhere(peakachuPreds.cpu().numpy() > 0.5).flatten()
            grouploopCalls = np.argwhere(grouploopPreds.cpu().numpy()>0.5).flatten()
            targetCalls = np.argwhere(targets.cpu().numpy() > 0.5).flatten()

            grouploopOnly = set(grouploopCalls)-set(targetCalls)#-set(peakachuCalls)
            pileup = np.zeros((21,21))
            for idx in grouploopOnly:
                pileup+=cmaps[idx,...]
            print(grouploopOnly)
            plt.figure()
            plt.imshow(pileup/len(grouploopOnly))
            plt.show()

            peakachuOnly = set(peakachuCalls)-set(targetCalls)#-set(grouploopCalls)
            pileup = np.zeros((21,21))
            for idx in peakachuOnly:
                pileup+=cmaps[idx,...]
            print(peakachuOnly)
            plt.figure()
            plt.imshow(pileup/len(peakachuOnly))
            plt.show()




            # plt.figure(figsize=(8, 8), dpi=300)
            venn3([set(grouploopCalls), set(peakachuCalls), set(targetCalls)], ('groupLoop', 'peakachu', 'target'))
            venn3_circles([set(grouploopCalls), set(peakachuCalls), set(targetCalls)])
            # plt.savefig('venn.eps')
            plt.show()


            for preds,tool in [(grouploopPreds,'grouploop'),(mlpPreds,'baseline'),(rfPreds,'RF'),(peakachuPreds,'peakachu')]:
                acc = torchmetrics.functional.accuracy(preds, targets.to(int))
                f1 = torchmetrics.functional.f1(preds, targets.to(int))
                auroc = torchmetrics.functional.auroc(preds, targets.to(int))
                precision, recall = torchmetrics.functional.precision_recall(preds, targets.to(int))

                print(key,tool,'===========================')
                print('acc=', acc)
                print('f1=', f1)
                print('precision=', precision)
                print('recall=', recall)
                print('auroc=', auroc)





if __name__ == '__main__':
    testAll()