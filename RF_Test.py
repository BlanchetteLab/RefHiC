import torch
import pandas as pd
import h5py
from gcooler import gcool
import click
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier as forest
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import gc
from sklearn.utils import shuffle
from joblib import dump, load
from tqdm import tqdm
import torchmetrics

def trainRF(x, y, nproc=10):
    """
    :param X: training set from buildmatrix
    :param distances:
    """
    print('#input data'.format(len(x)))
    gc.collect()
    params = {}
    params['class_weight'] = ['balanced', None]
    params['n_estimators'] = [50,100,200,300]
    params['n_jobs'] = [1]
    params['max_features'] = ['auto','sqrt','log2']
    params['max_depth'] = [20,50,100,150]
    params['random_state'] = [42]
    #from hellinger_distance_criterion import HellingerDistanceCriterion as hdc
    #h = hdc(1,np.array([2],dtype='int64'))
    params['criterion'] = ['gini','entropy']
    #model = forest(**params)
    mcc = metrics.make_scorer(metrics.matthews_corrcoef)
    model = GridSearchCV(forest(), param_grid=params,
                         scoring=mcc, verbose=2, n_jobs=nproc, cv=3)
    model.fit(x, y)
    fts = model.best_estimator_.feature_importances_[:]
    params = model.best_params_
    print(params)
    print(model.best_score_)
    return model.best_estimator_



import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@click.command()
@click.option('--trainingset', type=str, required=True, help='training data in .pkl or .h5 file; use if file existed; otherwise, prepare one for reuse')
@click.option('--skipchrom', type=str, default=None, help='skip one chromosome for during training')
@click.option('--resol', default=5000, help='resolution')
@click.option('--bedpe',type=str,default=None, help = '.bedpe file containing labelling cases')
@click.option('--test', type=str, default=None, help='comma separated test files in .gcool')
@click.option('--extra', type=str, default=None, help='a file contain a list of extra .gcools (i.e. database)')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1)")
@click.option('--oversampling',type=float, default = 1.0, help ='oversampling positive training cases, [0-2]')
@click.option('--feature',type = str, default = 0, help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--ti',type = int, default = None, help = 'use the indexed sample from the test group during training if multiple existed; values between [0,n)')
@click.option('--eval_ti',type = str,default = None, help = 'multiple ti during validating, ti,coverage; ti:coverage,...')
@click.option('--prefix',type=str,default='',help='output prefix')
@click.option('--savedModel',type=str,default=None,help='trained model')
def trainAttention(prefix, trainingset, skipchrom, resol, test, extra, bedpe, max_distance,w,oversampling,feature,ti,eval_ti,savedmodel):

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
        for i in tqdm(range(len(y_idx))):
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
            for i in range(len(X_test)):
                X_test[i] = X_test[i][:,featureMask]
                Xs_test[i] = Xs_test[i][:,featureMask]

                # Xs_test[i]=np.vstack([Xs_test[i],np.random.permutation(Xs_test[i].reshape(-1)).reshape(Xs_test[i].shape)])
                # print(Xs_test[i].shape)
            for i in range(len(X_val)):
                X_val[i] = X_val[i][:,featureMask]
                Xs_val[i] = Xs_val[i][:,featureMask]


    if eval_ti is None:
        eval_ti = {}
        for _i in range(X_train[0].shape[0]):
            eval_ti['sample'+str(_i)] = _i
    print('eval_ti',eval_ti)
    print('#train:',len(y_label_train))
    print('#test:', len(y_label_test))
    print('#validation:', len(y_label_val))

    if skipchrom is None:
        prefix = prefix+'_feature'+str(feature)+'_RF'
    else:
        prefix = prefix+'_feature'+str(feature) + '_chr' + str(skipchrom)+'_RF'

    for ti in range(X_test[0].shape[0]):
        print(ti,' ===============================')
        x_test=[]
        for i in range(len(X_test)):
            x_test.append(X_test[i][ti])

        RFModel = load(savedmodel)

        pred = RFModel.predict_proba(x_test)

        pred=torch.Tensor(pred)
        target = torch.Tensor(y_label_test)
        acc = torchmetrics.functional.accuracy(pred, target.to(int))
        f1 = torchmetrics.functional.f1(pred[:,1], target.to(int))
        auroc =torchmetrics.functional.auroc(pred[:,1], target.to(int))
        precision, recall = torchmetrics.functional.precision_recall(pred[:,1], target.to(int))

        print('acc=', acc)
        print('f1=',f1)
        print('precision=',precision)
        print('recall=',recall)
        print('auroc=', auroc)







if __name__ == '__main__':
    trainAttention()