import click
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pylab as plt
from scipy.sparse import coo_matrix
from refhic.util import fdr
import pandas as pd

def rhoDelta(data,resol,dc):
    pos = data[[1, 4]].to_numpy() // resol
    # remove singleton
    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(pos, r=2, return_distance=True)
    _l = []
    for v in NNindexes:
        _l.append(len(v))
    _l=np.asarray(_l)
    data = data[_l>5].reset_index(drop=True)
    # end of remove singleton
    pos = data[[1, 4]].to_numpy() // resol
    val = data[6].to_numpy()

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
        idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
        if idx.shape[0] == 0:
            deltas[i] = _dists[i][-1] + 1
        else:
            LargerNei[i] = _indexes[i][idx[0]]
            deltas[i] = _dists[i][idx[0]]
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

    data['rhos']=rhos
    data['deltas']=deltas

    return data


@click.command()
@click.option('--dc', type=int, default=25000, help='distance cutoff for local density calculation in terms of bp. default: 25000')
@click.option('--minscore', type=float,default=0.5, help='min RefHiC score')
@click.option('--resol', default=5000, help='resolution')
# @click.option('--interactive',default=False,type=bool,help='interactive mode for cutoff choosing')
@click.option('--alpha', type=float, default=0.05, help='FDR alpha [0.05]')
@click.option('--mindelta', type=float, default=5, help='min delta')
@click.option('--refine',type=bool,default = True,help ='refine')
@click.option('--verbose',type=bool,default =False, help='show plot')
@click.argument('candidates', type=str,required=True)
@click.argument('output', type=str,required=True)
def pool(dc,candidates,resol,mindelta,minscore,output,refine,alpha,verbose):
    '''call loop from loop candidates by clustering'''
    dc=dc/resol
    data = pd.read_csv(candidates, sep='\t', header=None)
    data = data[data[6] > minscore].reset_index(drop=True)
    data = data[data[4] - data[1] > 11*resol].reset_index(drop=True)
    data[['rhos','deltas']]=0
    data=data.groupby([0,9]).apply(rhoDelta,resol=resol,dc=dc).reset_index(drop=True)


    minrho=fdr(data[(data[9]=='target') & (data['deltas']>mindelta)]['rhos'],data[(data[9]=='decoy') & (data['deltas']>mindelta)]['rhos'],alpha=alpha)



    if verbose:
        plt.figure()
        plt.plot(data[(data[9]=='decoy') & (data['deltas']>mindelta)]['rhos'],data[(data[9]=='decoy')& (data['deltas']>mindelta)]['deltas'],'.',label='decoy')
        plt.plot(data[(data[9] == 'target')& (data['deltas']>mindelta)]['rhos'], data[(data[9] == 'target')& (data['deltas']>mindelta)]['deltas'],'.' ,label='target')
        plt.plot([minrho,minrho],[0,np.max(data['deltas'])])
        plt.legend()
        plt.show()

    targetData=data[data[9]=='target'].reset_index(drop=True)

    loopPds=[]
    for chrom in set(targetData[0]):
        data = targetData[targetData[0]==chrom].reset_index(drop=True)

        pos = data[[1, 4]].to_numpy() // resol
        posTree = KDTree(pos, leaf_size=30, metric='chebyshev')

        rhos = data['rhos'].to_numpy()
        deltas = data['deltas'].to_numpy()
        centroid = np.argwhere((rhos > minrho) & (deltas > mindelta)).flatten()


        # calculate delta_i, i.e. distance to nearest point with larger rho
        _r = 100
        _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
        LargerNei = rhos * 0 - 1
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
            if idx.shape[0] == 0:
                pass
            else:
                LargerNei[i] = _indexes[i][idx[0]]

        failed = np.argwhere(LargerNei == -1).flatten()
        while len(failed) > 1 and _r < 100000:
            _r = _r * 10
            _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
            for i in range(len(_indexes)):
                idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
                if idx.shape[0] == 0:
                    pass
                else:
                    LargerNei[failed[i]] = _indexes[i][idx[0]]
            failed = np.argwhere(LargerNei == -1).flatten()


        # assign rest loci to loop clusters
        LargerNei = LargerNei.astype(int)
        label = LargerNei * 0 - 1
        for i in range(len(centroid)):
            label[centroid[i]] = i
        decreasingsortedIdxRhos = np.argsort(-rhos)
        for i in decreasingsortedIdxRhos:
            if label[i] == -1:
                label[i] = label[LargerNei[i]]



        # refine loop
        val = data[6].to_numpy()
        refinedLoop = []
        label = label.flatten()
        for l in set(label):
            idx = np.argwhere(label == l).flatten()
            if len(idx) > 0:
                refinedLoop.append(idx[np.argmax(val[idx])])


        if refine:
            loopPds.append(data.loc[refinedLoop])
        else:
            loopPds.append(data.loc[centroid])

    loopPd=pd.concat(loopPds).sort_values(6,ascending=False)
    loopPd.to_csv(output,sep='\t',header=False, index=False)
    print(len(loopPd),'loops saved to ',output)

if __name__ == '__main__':
    pool()
