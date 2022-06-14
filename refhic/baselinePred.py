import torch
import pandas as pd
from torch.utils.data import DataLoader
from refhic.models import baselineNet,ensembles
import numpy as np
from bcooler import bcool
import click
from data import bcoolDataset,bedpewriter
from tqdm import tqdm


@click.command()
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--gpu', type=int, default=0, help='use GPU')
@click.option('--chrom', type=str, default=None, help='peaking calling for comma separated chroms')
@click.option('--resol', default=5000, help='resolution')
@click.option('--test', type=str, default=None, help='comma separated test files in .gcool')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1)")
@click.option('--encoding_dim',type = int, default =64,help='encoding dim')
@click.option('--feature',type = str, default = '1,2', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank')
@click.option('--modelState',type=str,default =None,help='trained model',required=True)
@click.option('--output',type=str,default = None,help ='output file name')
def baselinePred(batchsize, gpu, chrom, resol, test, max_distance,w,feature,encoding_dim,modelstate,output):
    if output is None:
        output = test +'.baseline_prob.bedpe'
    loopwriter = bedpewriter(output,resol)
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



        test_data = gcoolDataset(bmatrix,w,resol)


        test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False,num_workers=10,prefetch_factor=2)

        baselineModel = baseline(np.sum(featureMask), encoding_dim=encoding_dim).to(device)
        _modelstate = torch.load(modelstate, map_location='cuda:' + str(gpu))
        if 'model_state_dict' in _modelstate:
            _modelstate = _modelstate['model_state_dict']
        baselineModel.load_state_dict(_modelstate)
        baselineModel.eval()
        with torch.no_grad():
            for X in tqdm(test_dataloader):
                pred = torch.sigmoid(baselineModel(X[0][...,featureMask].to(device))).flatten().cpu()
                loop = np.argwhere(pred>0.5)
                prob = pred[loop].cpu().numpy().flatten().tolist()
                frag1 = X[1][0][loop].cpu().numpy().flatten().tolist()
                frag2 = X[1][1][loop].cpu().numpy().flatten().tolist()
                val=X[0][loop][:,:,(2*w+1)*w+w].numpy().flatten().tolist()
                p2ll=X[0][loop][:,:,(2 * w + 1) * (2 * w + 1) * 2 + (2 * w + 1) * 2 + 1].numpy().flatten().tolist()
                # print(p2ll)
                # print(X[0][loop].shape)
                loopwriter.write(_chrom,frag1,frag2,prob,val,p2ll)









if __name__ == '__main__':
    baselinePred()