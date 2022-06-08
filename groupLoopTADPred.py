import torch
import pandas as pd
from torch.utils.data import DataLoader
from groupLoopModels import attentionToAdditionalHiC,baseline,ensembles
import numpy as np
from gcooler import gcool
import click
from scipy.signal import find_peaks
from data import diagGcoolsDataset,bedwriter
from tqdm import tqdm


@click.command()
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--gpu', type=int, default=0, help='use GPU')
@click.option('--chrom', type=str, default=None, help='peaking calling for comma separated chroms')
@click.option('--resol', default=5000, help='resolution')
@click.option('-n', type=int, default=-1, help='sampling n samples from database; -1 for all')
@click.option('--test', type=str, default=None, help='comma separated test files in .gcool')
@click.option('--extra', type=str, default=None, help='a file contain a list of extra .gcools (i.e. database)')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1)")
@click.option('--cnn',type=bool,default=True,help='cnn encoder')
@click.option('--encoding_dim',type = int, default =64,help='encoding dim')
@click.option('--feature',type = str, default = '1,2,3,4,5', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--modelState',type=str,default =None,help='trained model',required=True)
@click.option('--output',type=str,default = None,help ='output file name')
def groupLoopPred(batchsize, gpu, chrom, resol, n, test, extra, max_distance,w,feature,encoding_dim,modelstate,output,cnn):
    if output is None:
        output = test +'.groupLoop_prob.bedpe'
    TADBwriter = bedwriter(output,resol)
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

    testGcool = gcool(test + '::/resolutions/' + str(resol))
    extraGcools = [gcool(file_path + '::/resolutions/' + str(resol)) for file_path in
                   pd.read_csv(extra, header=None)[0].to_list()]
    if chrom is None:
        chrom = ['chr'+str(i) for i in range(1,23)]
    else:
        if 'chr' in chrom:
            chrom = chrom.split(',')
        else:
            chrom = ['chr'+chr for chr in chrom.split(',')]

    for _chrom in chrom:
        print('analyzing chromosome ',_chrom,' ...')
        bmatrix = testGcool.bchr(_chrom, max_distance=max_distance)
        bmatrices = [x.bchr(_chrom,max_distance=max_distance) for x in extraGcools]

        if n == -1:
            test_data = diagGcoolsDataset(bmatrix,bmatrices,w,resol)
        else:
            test_data = diagGcoolsDataset(bmatrix, bmatrices, w, resol,samples=n)

        test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False,num_workers=10,prefetch_factor=2)

        if ';' in modelstate:
            savedmodel=modelstate.split(';')
            models=[]
            for _savemodel in savedmodel:
                model = attentionToAdditionalHiC(np.sum(featureMask), encoding_dim=encoding_dim,CNNencoder=cnn,win=2*w+1).to(
                    device)
                _modelstate = torch.load(_savemodel, map_location='cuda:' + str(gpu))
                if 'model_state_dict' in _modelstate:
                    _modelstate = _modelstate['model_state_dict']
                model.load_state_dict(_modelstate,strict=False)
                model.eval()
                models.append(model)
            model = ensembles(models)
        else:
            model = attentionToAdditionalHiC(np.sum(featureMask), encoding_dim=encoding_dim,CNNencoder=cnn,win=2*w+1).to(device)
            _modelstate = torch.load(modelstate, map_location='cuda:' + str(gpu))
            if 'model_state_dict' in _modelstate:
                _modelstate = _modelstate['model_state_dict']
            model.load_state_dict(_modelstate)
            model.eval()
        with torch.no_grad():
            for X in tqdm(test_dataloader):
                # X:X,Xs,Xcenter,yCenter
                pred = torch.sigmoid(model(X[0][...,featureMask].to(device),X[1][...,featureMask].to(device))).flatten().cpu()
                tad = np.argwhere(pred>=0)
                prob = pred[tad].cpu().numpy().flatten().tolist()

                targetPeaks, targetPeakProperties = find_peaks(prob, height=-9999, distance=2, plateau_size=0)

                TADs = np.zeros(len(prob))
                for j in range(len(targetPeaks)):
                    if targetPeakProperties['peak_heights'][j] >= 0.5:
                        TADs[targetPeaks[j]] = 1


                frag1 = X[2][0][tad].cpu().numpy().flatten().tolist()
                TADBwriter.write(_chrom,frag1,prob,TADs)









if __name__ == '__main__':
    groupLoopPred()