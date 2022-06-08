import cooler
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import sys
import random
from scipy import stats

cpath = sys.argv[1]#'/home/yanlin/workspace/PhD/nextProject/experiment/manuscript/draft_v1/downsample/4DNFIXP4QG5B_Rao2014_GM12878.mcool'
Lib = cooler.Cooler(cpath+'::/resolutions/5000')
target = sys.argv[2]#"../groupLoops/training-sets/gm12878.ctcf+h3k27ac_l20kb.hg38.bedpe"
# target =  pd.read_csv(target,header=None,sep='\t')


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
    for i in range(int(N*0.05)):
        r = np.random.random()
        ii = np.searchsorted(ref, r)
        part2.append(pool[ii])

    sample_dis = Counter(list(part1) + part2)
    sample_dis = Counter(list(part1))

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

def parsebed(chiafile, res=10000, lower=1, upper=5000000):

    coords = defaultdict(set)
    ambiguousXCoords = defaultdict(set)
    ambiguousYCoords = defaultdict(set)
    ambiguousXYCoords = defaultdict(set)
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
                elif a!=a2:
                    if b!=b2:
                        ambiguousXYCoords[chrom].add((a, b))
                    else:
                        ambiguousXCoords[chrom].add((a, b))
                elif b!=b2:
                    ambiguousYCoords[chrom].add((a, b))



    for c in coords:
        coords[c] = sorted(coords[c])
    for c in ambiguousXCoords:
        ambiguousXCoords[c] = sorted(ambiguousXCoords[c])
    for c in ambiguousYCoords:
        ambiguousYCoords[c] = sorted(ambiguousYCoords[c])
    for c in ambiguousXYCoords:
        ambiguousXYCoords[c] = sorted(ambiguousXYCoords[c])

    return coords,ambiguousXCoords,ambiguousYCoords,ambiguousXYCoords

coords,ambiguousXCoords,ambiguousYCoords,ambiguousXYCoords=parsebed(target,5000)


newCorrds = coords
kde, lower, long_start, long_end = learn_distri_kde(newCorrds)

all_neg_coords = {}

chromosomes=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22']
for key in chromosomes:
    if key.startswith('chr'):
        chromname = key
    else:
        chromname = 'chr' + key
    print('collecting from {}'.format(key))
    X = Lib.matrix(balance=True,
                   sparse=True).fetch(key).tocsr()

    clist = coords[chromname]
    neg_coords = negative_generating(
        X, kde, newCorrds[chromname], lower, long_start, long_end)

    all_neg_coords[chromname] = neg_coords


for key in newCorrds:
    for (x,y) in newCorrds[key]:
        print(key+'\t'+str(x*5000+1)+'\t'+str((x+1)*5000-1)+'\t'+key+'\t'+str(y*5000+1)+'\t'+str((y+1)*5000-1)+'\t1')

for key in all_neg_coords:
    for (x,y) in all_neg_coords[key]:
        print(key+'\t'+str(x*5000+1)+'\t'+str((x+1)*5000-1)+'\t'+key+'\t'+str(y*5000+1)+'\t'+str((y+1)*5000-1)+'\t0')
