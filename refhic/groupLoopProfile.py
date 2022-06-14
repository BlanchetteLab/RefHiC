import torch
import pandas as pd
from torch.utils.data import DataLoader
from groupLoopModels import attentionToAdditionalHiC,baseline,ensembles
import numpy as np
from gcooler import gcool
import click,sys
from data import gcoolsDataset,bedpewriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

@click.command()
@click.option('--gpu', type=int, default=0, help='use GPU')
@click.option('--resol', default=5000, help='resolution')
@click.option('--test', type=str, default=None, help='comma separated test files in .gcool')
@click.option('--extra', type=str, default=None, help='a file contain a list of extra .gcools (i.e. database)')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1)")
@click.option('--encoding_dim',type = int, default =64,help='encoding dim')
@click.option('--feature',type = str, default = '1,2', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--modelState',type=str,default =None,help='trained model',required=True)
@click.argument('A',type=str,default = None,required=True)
@click.argument('B',type=str,default = None,required=True)
@click.argument('output',type=str,default = None, required=True)
@click.option('--cnn',type=bool,default=True,help='cnn encoder')
def groupLoopProfile(gpu,cnn, a, b, resol, test, extra, w,feature,encoding_dim,modelstate,output,max_distance):
    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+ "cuda:"+str(gpu))
    else:
        device = torch.device("cpu")

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
    model = attentionToAdditionalHiC(np.sum(featureMask), encoding_dim=encoding_dim,CNNencoder=cnn,win=2*w+1).to(device)
    _modelstate = torch.load(modelstate, map_location='cuda:' + str(gpu))
    if 'model_state_dict' in _modelstate:
        _modelstate = _modelstate['model_state_dict']
        model.load_state_dict(_modelstate)
        model.eval()

    testGcool = gcool(test + '::/resolutions/' + str(resol))
    extraGcools = [gcool(file_path + '::/resolutions/' + str(resol)) for file_path in
                   pd.read_csv(extra, header=None)[0].to_list()]
    chrom = a.split(':')[0]
    if chrom!=b.split(':')[0]:
        print('invalid loop position .. ')
        sys.exit(0)
    posA=int(a.split(':')[-1])
    posB=int(b.split(':')[-1])
    if posA>posB:
        _tmp = posB
        posB = posA
        posA = _tmp

    bmatrix = testGcool.bchr(chrom, max_distance=max_distance)
    bmatrices = [x.bchr(chrom, max_distance=max_distance) for x in extraGcools]

    mat, meta = bmatrix.square(posA, posB, w, 'b', cache=False)
    X = np.concatenate((mat.flatten(), meta)).flatten()
    Xs = []
    for i in range(len(bmatrices)):
        mat, meta = bmatrices[i].square(posA, posB, w, 'b', cache=False)
        Xs.append(np.concatenate((mat.flatten(), meta)))
    Xs = np.vstack(Xs)[...,featureMask]
    X = X[featureMask]
    print(Xs.shape,X.shape,'.....')
    Xs = torch.from_numpy(Xs).float().to(device)
    X = torch.from_numpy(X).float().to(device)

    posTpl = torch.from_numpy(np.loadtxt('posVal.txt')).float()
    negTpl = torch.from_numpy(np.loadtxt('negVal.txt')).float()
    tpl = torch.cat([posTpl[None, :],negTpl[None, :]], dim=0)[...,featureMask].to(device)

    Xs=torch.cat([Xs,tpl],dim=0)


    with torch.no_grad():
        logit,alpha = model(X[None,...],Xs[None,...],returnAtten=True)

    X=X.cpu().numpy()
    Xs = Xs.cpu().numpy()
    print(X.shape,Xs.shape)
    print(torch.sigmoid(logit))
    alpha=alpha.detach().cpu().numpy().flatten()
    N = alpha.shape[0]
    win=2*w+1

    fig = plt.figure(figsize=(20,5))
    gs = GridSpec(nrows=3, ncols=N+1, height_ratios=[1,2,2])

    ax_ = fig.add_subplot(gs[1, 0])
    ax_.axes.xaxis.set_visible(False)
    ax_.axes.yaxis.set_visible(False)
    ax_.imshow(X[:win**2].reshape(win, win))

    ax_ = fig.add_subplot(gs[2, 0])
    ax_.axes.xaxis.set_visible(False)
    ax_.axes.yaxis.set_visible(False)
    ax_.imshow(X[win**2:win**2*2].reshape(win, win))

    for i in range(1,N+1):
        ax_ = fig.add_subplot(gs[0, i])
        ax_.bar(1, alpha[i-1], align='center', width=0.3)
        ax_.axes.xaxis.set_visible(False)
        ax_.spines['right'].set_visible(False)
        ax_.spines['top'].set_visible(False)
        ax_.axes.set_ylim([0,np.max(alpha)])


        ax_ = fig.add_subplot(gs[1, i])
        ax_.axes.xaxis.set_visible(False)
        ax_.axes.yaxis.set_visible(False)
        ax_.imshow(Xs[i-1, :win**2].reshape(win, win))

        ax_ = fig.add_subplot(gs[2, i])
        ax_.axes.xaxis.set_visible(False)
        ax_.axes.yaxis.set_visible(False)
        ax_.imshow(Xs[i-1, win**2:win**2*2].reshape(win, win))
    # plt.subplots_adjust(wspace=0.02, hspace=-0.6)
    plt.show()











if __name__ == '__main__':
    groupLoopProfile()