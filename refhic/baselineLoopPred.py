import torch
import pandas as pd
from torch.utils.data import DataLoader
from models import baselineNet,ensembles
import numpy as np
from bcooler import bcool
import click
from data import bcoolsDataset,bedpewriter
from tqdm import tqdm




@click.command()
@click.option('--batchsize', type=int, default=20480, help='batch size [20480]')
@click.option('--gpu', type=int, default=0, help='use GPU [0]')
@click.option('--chrom', type=str, default=None, help='loop  calling for comma separated chroms')
@click.option('-t', type=int, default=10, help='number of cpu threads; [10]')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('--modelState',type=str,default =None,required=True,help='trained model')
@click.argument('input', type=str,required=True)
@click.argument('output', type=str,required=True)
def pred(batchsize, gpu, chrom, input, max_distance,modelstate,output,t):
    '''Predict loop candidates from Hi-C contact map'''



    parameters = {}
    parameters['w'] = 10
    parameters['feature'] = '1,2'
    parameters['resol'] = 5000
    parameters['encoding_dim'] = 64
    parameters['cnn'] = True

    loopwriter = bedpewriter(output,parameters['resol'])
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
    extraBcools = [testBcool]
    if chrom is None:
        chrom =list(testBcool.chroms()['name'][:])
    else:
        if 'chr' in chrom:
            chrom = chrom.split(',')
        else:
            chrom = ['chr'+chr for chr in chrom.split(',')]

    if ';' in modelstate:
        modelStates=modelstate.split(';')
        models=[]
        for _modelState in modelStates:
            model = baselineNet(np.sum(featureMask), encoding_dim=parameters['encoding_dim'],CNNencoder=parameters['cnn'],win=2*parameters['w']+1).to(
                device)
            _modelstate = torch.load(_modelState, map_location='cuda:' + str(gpu))

            model.load_state_dict(_modelstate['model_state_dict'],strict=False)
            model.eval()
            models.append(model)
        model = ensembles(models)
    else:
        model = baselineNet(np.sum(featureMask), encoding_dim=parameters['encoding_dim'],CNNencoder=parameters['cnn'],win=2*parameters['w']+1).to(device)
        _modelstate = torch.load(modelstate, map_location='cuda:' + str(gpu))
        model.load_state_dict(_modelstate['model_state_dict'])
        model.eval()

    for _chrom in chrom:
        print('analyzing chromosome ',_chrom,' ...')
        bins = testBcool.bins().fetch(_chrom)
        weights={}
        for start,weight in zip(list(bins['start']),list(bins['weight'])):
            weights[start] = weight
            # print(start,weight)
        bad={}
        for start in weights:
            if np.isnan(weights[start]):
                bad[start]=True
            elif start-parameters['resol'] in weights and np.isnan(weights[start-parameters['resol']]):
                bad[start] = True
            elif start+parameters['resol'] in weights and np.isnan(weights[start+parameters['resol']]):
                bad[start] = True
            else:
                bad[start] = False

        bmatrix = [testBcool.bchr(_chrom, max_distance=max_distance,decoy=False),testBcool.bchr(_chrom, max_distance=max_distance,decoy=True,restrictDecoy=False)]
        bmatrices = [x.bchr(_chrom,max_distance=max_distance) for x in extraBcools]


        test_data = bcoolsDataset(bmatrix,bmatrices,parameters['w'],parameters['resol'])


        test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False,num_workers=t,prefetch_factor=4)


        with torch.no_grad():
            for X in tqdm(test_dataloader):
                # X:X,Xs,Xcenter,yCenter

                targetX=X[0][:,0,:]
                decoyX = X[0][:,1,:]
                # target
                pred = torch.sigmoid(model(targetX[...,featureMask].to(device))).flatten().cpu()
                loop = np.argwhere(pred>0.5).flatten()
                prob = pred[loop].cpu().numpy().flatten().tolist()
                frag1 = X[2][0][loop].cpu().numpy().flatten().tolist()
                frag2 = X[2][1][loop].cpu().numpy().flatten().tolist()
                isbad = [False] * len(frag1)
                for i in range(len(frag1)):
                    if bad[frag1[i]] or bad[frag2[i]]:
                        isbad[i] = True
                val=targetX[loop][:,(2*parameters['w']+1)*parameters['w']+parameters['w']].numpy().flatten().tolist()
                p2ll=targetX[loop][:,(2 * parameters['w'] + 1) * (2 * parameters['w'] + 1) * 2 + (2 * parameters['w'] + 1) * 2 + 1].numpy().flatten().tolist()
                loopwriter.write(_chrom,frag1,frag2,prob,val,p2ll,['target']*len(p2ll),isbad)
                # decoy
                pred = torch.sigmoid(model(decoyX[...,featureMask].to(device))).flatten().cpu()
                loop = np.argwhere(pred>0.5).flatten()
                prob = pred[loop].cpu().numpy().flatten().tolist()
                frag1 = X[2][0][loop].cpu().numpy().flatten().tolist()
                frag2 = X[2][1][loop].cpu().numpy().flatten().tolist()
                isbad = [False]*len(frag1)
                for i in range(len(frag1)):
                    if bad[frag1[i]] or bad[frag2[i]]:
                        isbad[i]=True
                val=decoyX[loop][:,(2*parameters['w']+1)*parameters['w']+parameters['w']].numpy().flatten().tolist()
                p2ll=decoyX[loop][:,(2 * parameters['w'] + 1) * (2 * parameters['w'] + 1) * 2 + (2 * parameters['w'] + 1) * 2 + 1].numpy().flatten().tolist()
                loopwriter.write(_chrom,frag1,frag2,prob,val,p2ll,['decoy']*len(p2ll),isbad=None)

if __name__ == '__main__':
    pred()