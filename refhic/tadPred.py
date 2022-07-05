import torch
import pandas as pd
from torch.utils.data import DataLoader
from refhic.models import refhicNet,baselineNet,ensembles
import numpy as np
from refhic.bcooler import bcool
import click
from scipy.signal import find_peaks
from refhic.data import diagBcoolsDataset,bedwriter
from tqdm import tqdm
from refhic.config import checkConfig,loadConfig,referenceMeta
import sys
from refhic.util import fdr


@click.command()
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--gpu', type=int, default=0, help='use GPU')
@click.option('--chrom', type=str, default=None, help='TAD boundary score calculation for comma separated chroms')
@click.option('-n', type=int, default=-1, help='sampling n samples from database; -1 for all')
@click.option('--reference', type=str, default=None, help='a file contains reference panel')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('--modelState',type=str,default =None,help='trained model')
@click.option('--alpha',type=float,default =0.05,help='FDR alpha')
@click.argument('input', type=str,required=True)
@click.argument('output', type=str,required=True)
def pred(batchsize, gpu, chrom, n, input, reference, max_distance,modelstate,output,alpha):
    '''Predict TAD boundary scores from Hi-C contact map'''
    if checkConfig():
        config=loadConfig()
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()

    reference = referenceMeta(reference)

    if modelstate is None:
        modelstate=config['tad']['model']
    parameters = torch.load(modelstate.split(';')[0],map_location='cuda:'+ str(gpu))['parameters']
    # print(parameters)
    print('***************************')
    if 'model' not in parameters:
        print('Model: unknown')
    else:
        print('Model:',parameters['model'])
    print('Resolution:',parameters['resol'])
    print('window size:',parameters['w']*2+1)
    print('***************************')



    if gpu is not None:
        device = torch.device("cuda:"+str(gpu))
        print('use gpu '+ "cuda:"+str(gpu))
    else:
        device = torch.device("cpu")
    _mask = np.zeros(2 * (parameters['w'] * 2 + 1) ** 2 + 2 * (2 * parameters['w'] + 1) + 4)
    featureMask = parameters['feature'].split(',')
    if '0' in featureMask:
        _mask[:] = 1
    if '1' in featureMask:
        _mask[:(2 * parameters['w'] + 1) ** 2] = 1
    if '2' in featureMask:
        _mask[(2 * parameters['w'] + 1) ** 2:2 * (2 * parameters['w'] + 1) ** 2] = 1
    if '3' in featureMask:
        _mask[2 * (2 * parameters['w'] + 1) ** 2:2 * (2 * parameters['w'] + 1) ** 2 + 2 * (2 * parameters['w'] + 1)] = 1
    if '4' in featureMask:
        _mask[2 * (2 * parameters['w'] + 1) ** 2 + 2 * (2 * parameters['w'] + 1)] = 1
    if '5' in featureMask:
        _mask[2 * (2 * parameters['w'] + 1) ** 2 + 2 * (2 * parameters['w'] + 1) + 1] = 1
    if '6' in featureMask:
        _mask[2 * (2 * parameters['w'] + 1) ** 2 + 2 * (2 * parameters['w'] + 1) + 2] = 1
    if '7' in featureMask:
        _mask[2 * (2 * parameters['w'] + 1) ** 2 + 2 * (2 * parameters['w'] + 1) + 3] = 1
    featureMask = np.ma.make_mask(_mask)


    testBcool = bcool(input + '::/resolutions/' + str(parameters['resol']))
    extraBcools = [bcool(file_path + '::/resolutions/' + str(parameters['resol'])) for file_path in
                   reference['file'].to_list()]
    if chrom is None:
        chrom = list(testBcool.chroms()['name'][:])
    else:
        if 'chr' in chrom:
            chrom = chrom.split(',')
        else:
            chrom = ['chr'+chr for chr in chrom.split(',')]

    result = {'chrom': [], 'start': [], 'end': [], 'lScore': [], 'lBoundary': []
        , 'rScore': [], 'rBoundary': [],'type':[]}
    for _chrom in chrom:
        print('analyzing chromosome ',_chrom,' ...')
        bmatrix = [testBcool.bchr(_chrom, max_distance=max_distance, decoy=False),
                   testBcool.bchr(_chrom, max_distance=max_distance, decoy=True,restrictDecoy=False)]

        bmatrices = [x.bchr(_chrom,max_distance=max_distance) for x in extraBcools]

        if n == -1:
            test_data = diagBcoolsDataset(bmatrix,bmatrices,parameters['w'],parameters['resol'])
        else:
            test_data = diagBcoolsDataset(bmatrix, bmatrices, parameters['w'], parameters['resol'],samples=n)

        test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False,num_workers=10,prefetch_factor=2)

        if ';' in modelstate:
            modelStates=modelstate.split(';')
            models=[]
            for _modelState in modelStates:
                model = refhicNet(np.sum(featureMask), encoding_dim=parameters['encoding_dim'],CNNencoder=parameters['cnn'],win=2*parameters['w']+1,
                                  classes=parameters['classes'],outputAct=torch.tanh).to(
                    device)
                _modelstate = torch.load(_modelState, map_location='cuda:' + str(gpu))
                model.load_state_dict( _modelstate['model_state_dict'],strict=False)
                model.eval()
                models.append(model)
            model = ensembles(models)
        else:
            model = refhicNet(np.sum(featureMask), encoding_dim=parameters['encoding_dim'],CNNencoder=parameters['cnn'],win=2*parameters['w']+1,
                              classes=parameters['classes'],outputAct=torch.tanh).to(device)
            _modelstate = torch.load(modelstate, map_location='cuda:' + str(gpu))
            model.load_state_dict(_modelstate['model_state_dict'])
            model.eval()
        with torch.no_grad():

            for X in tqdm(test_dataloader):
                # X:X,Xs,Xcenter,yCenter
                targetX=X[0][:,0,:]
                decoyX = X[0][:,1,:]

                # target
                pred = model(targetX[...,featureMask].to(device),X[1][...,featureMask].to(device)).cpu()
                predLs =pred[:,0].numpy().tolist()
                predRs = pred[:, 1].numpy().tolist()
                start = X[2][0].cpu().numpy().flatten().tolist()
                end = (X[2][0].cpu().numpy().flatten()+parameters['resol']).tolist()
                chroms = [_chrom]*len(start)
                result['chrom']+=chroms
                result['start']+=start
                result['end'] += end
                result['lScore']+= predLs
                result['lBoundary'] += [0]*len(start)
                result['rScore'] += predRs
                result['rBoundary'] += [0]*len(start)
                result['type'] += ['target'] * len(start)

                #decoy
                pred = model(decoyX[..., featureMask].to(device), X[1][..., featureMask].to(device)).cpu()
                predLs = pred[:, 0].numpy().tolist()
                predRs = pred[:, 1].numpy().tolist()
                start = X[2][0].cpu().numpy().flatten().tolist()
                end = (X[2][0].cpu().numpy().flatten() + parameters['resol']).tolist()
                chroms = [_chrom] * len(start)
                result['chrom'] += chroms
                result['start'] += start
                result['end'] += end
                result['lScore'] += predLs
                result['lBoundary'] += [0] * len(start)
                result['rScore'] += predRs
                result['rBoundary'] += [0] * len(start)
                result['type'] += ['decoy'] * len(start)


    result = pd.DataFrame.from_dict(result)
    for chrom in set(result['chrom']):
        lScore=result[(result['chrom']==chrom) & (result['type']=='target')]['lScore']
        targetPeaks, targetPeakProperties = find_peaks(lScore, height=-1, distance=5, plateau_size=0)
        lBoundary = np.zeros(len(lScore))

        decoyLScore=result[(result['chrom']==chrom) & (result['type']=='decoy')]['lScore']
        decoyPeaks, decoyPeakProperties = find_peaks(decoyLScore, height=-1, distance=5, plateau_size=0)

        heightCutoff = fdr(targetPeakProperties['peak_heights'], decoyPeakProperties['peak_heights'], alpha=alpha)
        # print(heightCutoff)
        for j in range(len(targetPeaks)):
            if targetPeakProperties['peak_heights'][j] >= heightCutoff:
                lBoundary[targetPeaks[j]] = 1

        rScore = result[(result['chrom']==chrom) & (result['type']=='target')]['rScore']
        targetPeaks, targetPeakProperties = find_peaks(rScore, height=-1, distance=5, plateau_size=0)
        rBoundary = np.zeros(len(rScore))

        decoyRScore=result[(result['chrom']==chrom) & (result['type']=='decoy')]['rScore']
        decoyPeaks, decoyPeakProperties = find_peaks(decoyRScore, height=-1, distance=5, plateau_size=0)

        heightCutoff = fdr(targetPeakProperties['peak_heights'], decoyPeakProperties['peak_heights'], alpha=alpha)
        # print(heightCutoff)

        for j in range(len(targetPeaks)):
            if targetPeakProperties['peak_heights'][j] >= heightCutoff:
                rBoundary[targetPeaks[j]] = 1
        result.loc[(result['chrom']==chrom) & (result['type']=='target'),'lBoundary']=np.asarray(lBoundary)
        result.loc[(result['chrom']==chrom) & (result['type']=='target'),'rBoundary']=np.asarray(rBoundary)
        print('np.sum(rBoundary)', np.sum(rBoundary))
        print('np.sum(lBoundary)', np.sum(lBoundary))
    # result=result[result['type']=='target'].reset_index(drop=True)
    result.to_csv(output,sep='\t',index=False,header=False)


    from matplotlib import pylab as plt
    data = pd.read_csv(output, sep='\t', header=None)

    for chrom in set(data[0]):
        x=data[data[0]==chrom]
        plt.figure()
        plt.plot(x[x[7] == 'target'][1] / 5000, x[x[7] == 'target'][3], label='target')
        plt.plot(x[x[7] == 'decoy'][1] / 5000, x[x[7] == 'decoy'][3], label='decoy')
        plt.legend()
        plt.title(chrom)
        plt.show()



if __name__ == '__main__':
    pred()