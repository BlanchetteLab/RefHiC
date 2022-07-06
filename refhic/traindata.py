import torch
import pandas as pd
import numpy as np
from refhic.bcooler import bcool
import click
import pickle
from refhic.config import checkConfig
import sys
from refhic.config import referenceMeta


@click.command()
@click.option('--resol', default=5000, help='resolution [5000]')
@click.option('--reference', type=str, default=None, help='a file contains reference panel')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=10, help="peak window size: (2w+1)x(2w+1) [10]")
@click.option('--feature',type = str, default = '1,2', help = 'a list of comma separated features: 0: all features; 1: contact map; 2: distance normalized contact map;'
                                                          '3: bias; 4: total RC; 5: P2LL; 6: distance; 7: center rank  [1,2]')
@click.argument('foci', type=str,required=True)
@click.argument('study', type=str,required=True)
@click.argument('output', type=str,required=True)
def traindata(output, resol,study, reference, foci, max_distance,w,feature):
    """Create train data from contact maps and labelled foci

    \b
    foci: contact pair with labels for the study sample
    study: comma separated mcool or bcool files (should be downsampled files) for the study sample
    output: output file prefix
    \b
    foci format:
        chr1 bin1_start bin1_end chr2 bin2_start bin2_end class_1 <class_2 ...>
    """

    if checkConfig():
        pass
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()

    dataParams={'resol':resol,'feature':feature,'w':w}
    reference = referenceMeta(reference)

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



    studyBcools  = [bcool(file_path+'::/resolutions/'+str(resol)) for file_path in study.split(',')]
    referencedBcools = [bcool(file_path+'::/resolutions/'+str(resol)) for file_path in reference['file'].to_list()]
    _foci = pd.read_csv(foci, header=None, sep='\t')

    # read labels
    labels = {}

    for row in _foci.itertuples():
        chr1 = row[1]
        pos1 = row[2]
        chr2 = row[4]
        pos2 = row[5]
        label = list(row[7:])
        if chr1 == chr2 and abs(pos2 - pos1) < max_distance:
            if chr1 not in labels:
                labels[chr1] = {'contact':[],'label':[]}
            labels[chr1]['contact'].append((pos1,pos2))
            labels[chr1]['label'].append(label)


    dataParams['classes'] = len(label)
    dataParams['featureDim'] = np.sum(featureMask)
    print('This training data contains ', dataParams['featureDim'], ' features, and ', dataParams['classes'], ' targets per item')


    Xs = {} # for extra
    label={}
    X = {}
    for chrom in labels:
        if chrom not in X:
            label[chrom] = []
        for i in range(len(labels[chrom]['label'])):
            label[chrom].append(labels[chrom]['label'][i])

        for b in studyBcools:
            if chrom not in X:
                X[chrom] = [[] for _ in range(len(labels[chrom]['contact']))]
            bmatrix = b.bchr(chrom, max_distance=max_distance)
            for i in range(len(labels[chrom]['contact'])):
                x,y = labels[chrom]['contact'][i]

                mat,meta = bmatrix.square(x,y,w,'b')
                X[chrom][i].append(np.concatenate((mat.flatten(), meta)))

        for b in referencedBcools:
            if chrom not in Xs:
                Xs[chrom] = [[] for _ in range(len(labels[chrom]['contact']))]
            bmatrix = b.bchr(chrom, max_distance=max_distance)
            for i in range(len(labels[chrom]['contact'])):
                x,y = labels[chrom]['contact'][i]
                mat,meta = bmatrix.square(x,y,w,'b')
                Xs[chrom][i].append(np.concatenate((mat.flatten(), meta)))


    X_train = []
    Xs_train = []
    y_label_train = []
    X_test = []
    Xs_test = []
    y_label_test = []
    X_val = []
    Xs_val = []
    y_label_val = []
    for chrom in X:
        for i in range(len(X[chrom])):
            x=np.asarray(X[chrom][i])[:, featureMask]
            xs=np.asarray(Xs[chrom][i])[:, featureMask]
            if chrom in chromTest:
                X_test.append(x)
                Xs_test.append(xs)
                y_label_test.append(label[chrom][i])
            elif chrom in chromVal:
                X_val.append(x)
                Xs_val.append(xs)
                y_label_val.append(label[chrom][i])
            else:
                X_train.append(x)
                Xs_train.append(xs)
                y_label_train.append(label[chrom][i])
    del X,Xs
    with open(output+'.pkl', 'wb') as handle:
        pickle.dump((dataParams,X_train, Xs_train, y_label_train,
                 X_test, Xs_test, y_label_test,
                 X_val, Xs_val, y_label_val), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    traindata()
