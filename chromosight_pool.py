import click
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pylab as plt
from scipy.sparse import coo_matrix
import pandas as pd
from chromosight_util import pick_foci
@click.command()
@click.option('--dc', type=int, default=25000, help='distance cutoff for local density calculation in terms of bp. default: 25000')
@click.option('--prob', type=str,required=True, help='grouploop probs')
@click.option('--minprob', type=float,default=0.5, help='min grouploop probs')
@click.option('--resol', default=5000, help='resolution')
@click.option('--interactive',default=True,type=bool,help='interactive mode for cutoff choosing')
@click.option('--minrho', type=float, default=10, help='min rho')
@click.option('--mindelta', type=float, default=5, help='min delta')
@click.option('--output',type=str,default = None,help ='output file name')
@click.option('--refine',type=bool,default = True,help ='refine')
def pool(dc,prob,resol,interactive,minrho,mindelta,minprob,output,refine):
    dc=dc/resol
    data = pd.read_csv(prob, sep='\t', header=None)
    pos = data[[1, 4]].to_numpy() // 5000
    val = data[6].to_numpy()#*data[7].to_numpy()

    coo = coo_matrix((val, (pos[:, 0].astype(int), pos[:, 1].astype(int))), dtype=float)
    from chromosight_util import pick_foci
    coords, _ = pick_foci(coo, minprob)
    # remove singleton
    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindex, NNdists = posTree.query_radius(coords, r=0, return_distance=True)
    centroid=[]
    for p in NNindex:
        centroid.append(p[0])
    centroid=np.asarray(centroid)



    plt.figure()
    coo = coo_matrix((val, (pos[:, 0].astype(int), pos[:, 1].astype(int))), dtype=float)
    plt.imshow(coo.toarray())
    plt.scatter(pos[centroid][:, 1], pos[centroid][:, 0], label='propose')
    target = pd.read_csv('chr17_target.bedpe', header=None, sep='\t')
    tpos = (target[[1, 4]] // 5000).to_numpy()
    plt.scatter(tpos[:, 1], tpos[:, 0], label='Target', facecolors='black', edgecolors='none')

    correct = np.array([x for x in set(tuple(x) for x in tpos) & set(tuple(x) for x in pos[centroid])])
    plt.scatter(correct[:, 1], correct[:, 0], label='match',color='red')
    plt.title(str(prob)[:10]+','+str(len(correct)))
    plt.legend()
    plt.xlim([500, 9500])
    plt.ylim([500, 9500])
    plt.gca().invert_yaxis()
    plt.show()

    print('find ',len(centroid),'loops..\n save loop to ',output)
    data.loc[centroid].to_csv(output,sep='\t',header=False, index=False)

if __name__ == '__main__':
    pool()