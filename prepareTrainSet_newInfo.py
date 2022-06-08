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

class loopDataset(Dataset):
    def __init__(self, extrafiles, testfile, positions, labels, win=5, resol=5000, samples=None, transform=None, target_transform=None):
        self.extraMats = []
        self.extraCools = []
        for extrafile in extrafiles:
            # print(extrafile + '::/resolutions/'+str(resol))
            self.extraMats.append(
                cooler.Cooler(extrafile + '::/resolutions/'+str(resol)).matrix(balance=True)
                                   )
            self.extraCools.append(cooler.Cooler(extrafile + '::/resolutions/'+str(resol)))
        self.testMat = cooler.Cooler(testfile + '::/resolutions/'+str(resol)).matrix(balance=True)
        self.testCool = cooler.Cooler(testfile + '::/resolutions/'+str(resol))
        self.positions = positions
        self.labels = torch.from_numpy(labels).float()
        self.transform = transform
        self.samples = samples
        self.win = win*resol
        self.resol =resol
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]

    def individualData(self,coolMat,coolFile,regionX,regionY,distance):
        mat = np.nan_to_num(coolMat.fetch(regionX,regionY))
        l = mat.shape[0]
        p2ll = mat[l//2,l//2]/(np.mean(mat[-(l//4):,:l//4])+1)
        mat = mat.flatten()
        mat = np.hstack((mat,
                         coolFile.bins()['weight'].fetch(regionX).to_numpy(),
                         coolFile.bins()['weight'].fetch(regionY).to_numpy(),
                         coolFile.open()['bins/weight'].attrs['scale'],coolFile.info['sum'],p2ll))
        # print('mat.shape',mat.shape)
        return np.nan_to_num(mat)
        # ranks = stats.rankdata(mat, method='average')
        # return np.hstack((mat, ranks))

    def __getitem__(self, idx):
        y = self.labels[idx]
        chrom = str(self.positions[idx]['chrom'])
        pos1 = int(self.positions[idx]['pos1'])
        pos2 = int(self.positions[idx]['pos2'])
        distance = abs(pos1-pos2)//self.resol
        regionX = chrom+':'+str(pos1 - self.win) + '-' + str(pos1 + self.win + 1)
        regionY = chrom+':'+str(pos2 - self.win) + '-' + str(pos2 + self.win + 1)
        # print(regionX,regionY)
        if self.samples:
            selectedIdx = np.random.choice(len(self.extraMats), self.samples, replace=False)
            X = self.individualData(self.extraMats[selectedIdx[0]],self.extraCools[[selectedIdx[0]]],regionX,regionX,distance)
            for i in selectedIdx[1:]:
                x=self.individualData(self.extraMats[i],self.extraCools[i],regionX,regionY,distance)
                X=np.vstack((x,X))
        else:
            X = self.individualData(self.extraMats[0],self.extraCools[0],regionX,regionX,distance)
            for i in range(1,len(self.extraMats)):
                x=self.individualData(self.extraMats[i],self.extraCools[i],regionX,regionY,distance)
                X=np.vstack((x,X))
        testData=self.individualData(self.testMat,self.testCool,regionX,regionY,distance)
        np.nan_to_num(X, copy=False)
        np.nan_to_num(testData, copy=False)
        testData=torch.from_numpy(testData).float()

        X = torch.from_numpy(X).float()
        X=(testData,X,regionX,regionY)

        if self.transform:
            return self.transform(X), y
        return X,y




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

def learn_distri_kde(coords):

    dis = []
    for c in coords:
        for a, b in coords[c]:
            dis.append(b-a)

    lower = min(dis)

    # part 1: same distance distribution as the positive input
    kde = stats.gaussian_kde(dis)

    # part 2: random long-range interactions
    counts, bins = np.histogram(dis, bins=100)
    long_end = int(bins[-1])
    tp = np.where(np.diff(counts) >= 0)[0] + 2
    long_start = int(bins[tp[0]])

    return kde, lower, long_start, long_end

def negative_generating(M, kde, positives, lower, long_start, long_end):

    positives = set(positives)
    N = 3 * len(positives)
    # part 1: kde trained from positive input
    part1 = kde.resample(N).astype(int).ravel()
    part1 = part1[(part1 >= lower) & (part1 <= long_end)]

    # part 2: random long-range interactions
    part2 = []
    pool = np.arange(long_start, long_end+1)
    tmp = np.cumsum(M.shape[0]-pool)
    ref = tmp / tmp[-1]
    for i in range(N):
        r = np.random.random()
        ii = np.searchsorted(ref, r)
        part2.append(pool[ii])

    sample_dis = Counter(list(part1) + part2)

    shifts = [-3,-2,-1,1,2,3]
    neg_coords = []
    # for p in positives:
    #     p1 = p[0] +random.choice(shifts)
    #     p2 = p[1] + random.choice(shifts)
    #     if p1>p2:
    #         neg_coords.append((p2,p1))
    #     else:
    #         neg_coords.append((p1,p2))
    # neg_coords = list(set(neg_coords)-positives)

    midx = np.arange(M.shape[0])
    for i in sorted(sample_dis):  # i cannot be zero
        n_d = sample_dis[i]
        R, C = midx[:-i], midx[i:]
        tmp = np.array(M[R, C]).ravel()
        tmp[np.isnan(tmp)] = 0
        mask = tmp > 0
        R, C = R[mask], C[mask]
        pool = set(zip(R, C)) - positives
        sub = random.sample(pool, n_d)
        neg_coords.extend(sub)

    random.shuffle(neg_coords)

    return neg_coords

def prepareData(chrom,pos, neg,resolution,testCool,outputFile,win):
    """
    :param X: training set from buildmatrix
    :param distances:
    """
    print('chrom:',chrom)

    indices = np.random.choice(neg.shape[0], pos.shape[0], replace=False)
    neg = neg[indices,:]
    print('pos.shape,neg.shape', pos.shape, neg.shape)
    X=[]
    y=[]
    for i in range(len(pos)):
        X.append({'chrom':chrom,'pos1':pos[i,0],'pos2':pos[i,1]})
        y.append(1)
    for i in range(len(neg)):
        X.append({'chrom':chrom,'pos1':neg[i,0],'pos2':neg[i,1]})
        y.append(0)
    X=np.asarray(X)
    y=np.asarray(y)


    f = pd.read_csv('./smalldb.txt',header=None)
    extrafiles = list(f[0])
    print('number of samples in database:',len(extrafiles))
    testfile = testCool
    test_data = loopDataset(extrafiles,testfile,X,y,resol=resolution,win=win)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=True,num_workers=10)
    idx = outputFile.attrs.get('samples')
    j=0
    for data, target in test_dataloader:
        for i in range(len(data[0])):
            # print(j)
            outputFile.create_dataset(chrom+'/'+str(idx)+'/test',data=data[0][i].numpy())
            outputFile.create_dataset(chrom+'/'+str(idx)+'/db', data=data[1][i].numpy())
            outputFile[chrom+'/'+str(idx)].attrs['frag1']=data[-2][i]
            outputFile[chrom+'/'+str(idx)].attrs['frag2']=data[-1][i]
            outputFile[chrom+'/'+str(idx)].attrs['target']=int(target[i])
            j=j+1
            idx=idx+1
        outputFile.flush()
    outputFile.attrs['samples']=idx
    outputFile.flush()



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


    prepareData(chromname,positiveLoops*args.resolution,negativeLoops*args.resolution,args.resolution,path,outputFile,win=args.width)
