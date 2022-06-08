import torch,sys
from torch.utils.data import DataLoader
from util import update_bn
from groupLoopModels import attentionToAdditionalHiC
import numpy as np
import click
from data import  inMemoryDataset
import pickle
import random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@click.command()
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--gpu', type=int, default=0, help='GPU training')
@click.option('--trainingset', type=str, required=True, help='training data in .pkl or .h5 file; use if file existed; otherwise, prepare one for reuse')
@click.option('-n', type=int, default=10, help='sampling n samples from database; -1 for all')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1)")
@click.option('--encoding_dim',type = int, default =64,help='encoding dim')
@click.option('--feature',type = str, default = '1,2,3,4,5', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')

@click.option('--states',type=str,default =None,required=True,help='model states: state1;state2;..;..')
@click.option('--prefix',type=str,default='',help='output prefix')
@click.option('--cnn',type=bool,default=True,help='cnn encoder')
@click.option('--lm',type=bool,default=True,help='large memory')

def modelAverage(lm,cnn,prefix,batchsize, gpu, trainingset, n,w,feature,encoding_dim,states):
    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+"cuda:"+str(gpu))
    else:
        device = torch.device("cpu")
    if lm:
        occccccc = torch.zeros((256,1024,18000)).to(device)

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





    if trainingset.endswith('.pkl'):
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
            for i in range(len(X_val)):
                X_val[i] = X_val[i][:,featureMask]
                Xs_val[i] = Xs_val[i][:,featureMask]
    else:
        print('you need to provide .pkl training data')
        sys.exit(0)



    prefix = prefix+'_feature'+str(feature)

    print('#training cases',len(y_label_train))
    if n == -1:
        n = None

    training_data = inMemoryDataset(X_train,Xs_train,y_label_train,samples=n)


    randGen = torch.Generator()
    randGen.manual_seed(42)
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True,num_workers=1,worker_init_fn = seed_worker)



    model = attentionToAdditionalHiC(np.sum(featureMask),encoding_dim=encoding_dim,CNNencoder=cnn).to(device)

    pts=states.split(';')
    avgModelState = torch.load(pts[0], map_location='cuda:'+str(gpu))
    if 'model_state_dict' in avgModelState:
        avgModelState = avgModelState['model_state_dict']
    for i in range(1, len(pts)):
        modelstate = torch.load(pts[i], map_location='cuda:'+str(gpu))
        if 'model_state_dict' in modelstate:
            modelstate = modelstate['model_state_dict']
            for key in avgModelState:
                avgModelState[key] += modelstate[key]
    for key in avgModelState:
        avgModelState[key] = avgModelState[key] / len(pts)
    model.load_state_dict(avgModelState)
    torch.save(model.state_dict(), prefix + '_groupLoop_WithoutBNupdate.pt')
    update_bn(train_dataloader, model,device=device)
    torch.save(model.state_dict(), prefix + '_groupLoop_BNupdated.pt')


if __name__ == '__main__':
    modelAverage()
