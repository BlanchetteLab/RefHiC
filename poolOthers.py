import click
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pylab as plt
from scipy.sparse import coo_matrix
import pandas as pd
@click.command()
@click.option('--dc', type=int, default=25000, help='distance cutoff for local density calculation in terms of bp. default: 25000')
@click.option('--prob', type=str,required=True, help='grouploop probs')
@click.option('--resol', default=5000, help='resolution')
@click.option('--interactive',default=False,type=bool,help='interactive mode for cutoff choosing')
@click.option('--minrho', type=float, default=10, help='min rho')
@click.option('--mindelta', type=float, default=5, help='min delta')
@click.option('--output',type=str,default = None,help ='output file name')
@click.option('--refine',type=bool,default = True,help ='refine')
@click.option('--chrom',type=str,default = None,help ='chrom')
def pool(dc,prob,resol,interactive,minrho,mindelta,output,refine,chrom):
    dc=dc/resol

    data = pd.read_csv(prob, sep='\t', header=None)
    data = data[data[0]==data[3]]
    chrom=int(chrom)
    data=data[data[0]==chrom].reset_index(drop=True)


    pos = data[[1, 4]].to_numpy() // 5000
    val = data[1].to_numpy()*0+1#*data[7].to_numpy()





    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(pos, r=dc, return_distance=True)

    # calculate local density rho
    rhos = []
    for i in range(len(NNindexes)):
        # rhos.append(np.sum(np.exp(-(NNdists[i] / dc) ** 2)))
        rhos.append(np.dot(np.exp(-(NNdists[i] / dc) ** 2), val[NNindexes[i]]))
    rhos = np.asarray(rhos)


    # calculate delta_i, i.e. distance to nearest point with larger rho
    _r = 100
    _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
    deltas = rhos * 0
    LargerNei = rhos * 0 - 1
    for i in range(len(_indexes)):
        idx = np.argwhere(rhos[_indexes[i]] >= rhos[_indexes[i][0]])
        if idx.shape[0] == 1:
            deltas[i] = _dists[i][-1] + 1
        else:
            LargerNei[i] = _indexes[i][idx[1]]
            deltas[i] = _dists[i][idx[1]]
    failed = np.argwhere(LargerNei == -1).flatten()
    while len(failed) > 1 and _r < 100000:
        _r = _r * 10
        _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
            if idx.shape[0] == 0:
                deltas[failed[i]] = _dists[i][-1] + 1
            else:
                LargerNei[failed[i]] = _indexes[i][idx[0]]
                deltas[failed[i]] = _dists[i][idx[0]]
        failed = np.argwhere(LargerNei == -1).flatten()

    # select draf loop
    if interactive:
        print('Please choose min rho and min delta ...')
        plt.figure()
        plt.plot(rhos, deltas, '.')
        plt.xlabel('rho')
        plt.ylabel('delta')
        plt.yscale('log')
        plt.show()
        minrho = float(input("Enter min rho:"))
        print("min rho is: ", minrho)
        mindelta = float(input("Enter min delta:"))
        print("min delta is: ", mindelta)
    centroid = np.argwhere((rhos > minrho) & (deltas > mindelta)).flatten()




    print('....',len(centroid))






    print('find ',len(centroid),'loops..\n save loop to ',output)
    data.loc[centroid].to_csv(output,sep='\t',header=False, index=False)

if __name__ == '__main__':
    pool()
