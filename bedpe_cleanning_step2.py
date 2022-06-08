import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from torch.utils.data import Dataset,DataLoader
import cooler
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import random
import h5py


def parsebed(chiafile, res=10000, lower=1, upper=5000000):

    coords = defaultdict(set)
    ambiguousCoords = defaultdict(set)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, a2, b, b2 = float(s[1]), float(s[2]),float(s[4]),float(s[5])
            a, b = int(a), int(b)
            a2, b2 = int(a2), int(b2)
            if a > b:
                a, b = b, a
                a2, b2= b2, a2
            a //= res
            a2 //= res
            b //= res
            b2 //= res
            # all chromosomes including X and Y
            if (b-a > lower) and (b-a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + s[0].lstrip('chr')
                if a==a2 and b==b2:
                    coords[chrom].add((a, b))
                    ambiguousCoords[chrom].add((a, b))
                elif a!=a2:
                    if b!=b2:
                        ambiguousCoords[chrom].add((a, b))
                        ambiguousCoords[chrom].add((a, b2))
                        ambiguousCoords[chrom].add((a2, b))
                        ambiguousCoords[chrom].add((a2, b2))
                    else:
                        ambiguousCoords[chrom].add((a, b))
                        ambiguousCoords[chrom].add((a2, b))
                elif b!=b2:
                    ambiguousCoords[chrom].add((a, b))
                    ambiguousCoords[chrom].add((a, b2))


    for c in coords:
        coords[c] = sorted(coords[c])
    for c in ambiguousCoords:
        ambiguousCoords[c] = sorted(ambiguousCoords[c])

    return coords,ambiguousCoords








import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bedpe")
parser.add_argument('--out',type=str)
parser.add_argument("--resolution",type=int)
parser.add_argument('--width',type=int)
path = '../../nextProject/experiment/manuscript/draft_v1/downsample/4DNFIXP4QG5B_Rao2014_GM12878.mcool'
args = parser.parse_args()
Lib = cooler.Cooler(path+'::/resolutions/'+str(args.resolution))
ofile = args.out
coords,ambiguousCoords = parsebed(args.bedpe, lower=2, res=args.resolution)

kde, lower, long_start, long_end = learn_distri_kde(coords)



positive_class = {}
negative_class = {}
chromosomes=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22']
all_neg_coords = {}
maxCoords = {}
all_pos_coords = coords
for key in chromosomes:
    if key.startswith('chr'):
        chromname = key
    else:
        chromname = 'chr'+key
    print('collecting from {}'.format(key))
    X = Lib.matrix(balance=True,
        sparse=True).fetch(key).tocsr()
        
    clist = coords[chromname]
    neg_coords = negative_generating(
        X, kde, ambiguousCoords[chromname], lower, long_start, long_end)
    
    maxCoords[chromname] = X.shape[0]
    all_neg_coords[chromname] =neg_coords
        

outputFile=h5py.File(ofile, 'w')

if outputFile.attrs.get('samples'):
    pass
else:
    outputFile.attrs['samples'] = 0

if outputFile.attrs.get('win'):
    if outputFile.attrs.get('win') != args.width:
       print('win should be ',outputFile.attrs.get('win'))
else:
    outputFile.attrs['win'] = args.width
#
# for key in all_neg_coords:
#     print(key,'....',all_neg_coords[key])
for key in chromosomes:
    if key.startswith('chr'):
        chromname = key
    else:
        chromname = 'chr'+key
    if len(all_pos_coords[chromname]) == 0:
        continue
    positiveLoops = np.asarray(all_pos_coords[chromname])
    negativeLoops = np.asarray(all_neg_coords[chromname])

    positiveLoops = positiveLoops[(positiveLoops[:, 0] > 2*args.width) & (positiveLoops[:, 1]  > 2*args.width)
                                    & (positiveLoops[:, 0] < maxCoords[chromname] -args.width -1)
                                    & (positiveLoops[:, 1] < maxCoords[chromname] -args.width -1), :]
    negativeLoops = negativeLoops[(negativeLoops[:, 0] > 2*args.width) & (negativeLoops[:, 1]  > 2*args.width)
                                    & (negativeLoops[:, 0] < maxCoords[chromname] -args.width -1)
                                    & (negativeLoops[:, 1] < maxCoords[chromname] -args.width -1), :]
    # print('positiveLoops.shape,negativeLoops.shape', positiveLoops.shape, negativeLoops.shape)



