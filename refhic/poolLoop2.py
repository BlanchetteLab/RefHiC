import click
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pylab as plt
from scipy.sparse import coo_matrix
import pandas as pd


def compute_cluster_stats(df):
    df['cluster_size'] = df.shape[0]
    df['neg_log10_fdr'] = np.sum(-np.log10(df['fdr_dist']))
    #df['summit'] = 0
    #df.loc[df['fdr_dist'].idxmin(),'summit'] = 1
    return df


@click.command()
@click.option('--dc', type=int, default=25000, help='distance cutoff for local density calculation in terms of bp. default: 25000')
@click.option('--minprob', type=float,default=0.5, help='min grouploop probs')
@click.option('--resol', default=5000, help='resolution')
@click.option('--interactive',default=False,type=bool,help='interactive mode for cutoff choosing')
@click.option('--minrho', type=float, default=10, help='min rho')
@click.option('--mindelta', type=float, default=5, help='min delta')
@click.option('--refine',type=bool,default = True,help ='refine')
@click.option('--chrom',type=str,default = None,help ='chrom')
@click.argument('candidates', type=str,required=True)
@click.argument('output', type=str,required=True)
def pool2(dc,candidates,resol,interactive,minrho,mindelta,minprob,output,refine,chrom):
    '''call loop from loop candidates by clustering'''
    dc=dc/resol
    data = pd.read_csv(candidates, sep='\t', header=None)
    data = data[data[6] > minprob].reset_index(drop=True)
    data = data[data[4] - data[1] > 11*resol].reset_index(drop=True)
    pos = data[[1, 4]].to_numpy() // resol
    val = data[6].to_numpy()#*data[7].to_numpy()

    # remove singleton
    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(pos, r=2, return_distance=True)
    _l = []
    for v in NNindexes:
        _l.append(len(v))
    _l=np.asarray(_l)
    data = data[_l>5].reset_index(drop=True)
    pos = data[[1, 4]].to_numpy() // 5000
    val = data[6].to_numpy()#*data[7].to_numpy()
    # val = data[7].to_numpy()
    # val = data[6].to_numpy()*data[7].to_numpy()
    # end of


    posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(pos, r=dc, return_distance=True)

    # calculate local density rho
    rhos = []
    for i in range(len(NNindexes)):
        # rhos.append(np.sum(np.exp(-(NNdists[i] / dc) ** 2)))
        rhos.append(np.dot(np.exp(-(NNdists[i] / dc) ** 2), val[NNindexes[i]]))
    rhos = np.asarray(rhos)
    # print(rhos)

    # rows = {}
    # cols = {}
    # for i in range(val.shape[0]):
    #     if pos[i, 0] not in rows:
    #         rows[pos[i, 0]] = {'col': [], 'val': []}
    #     if pos[i, 1] not in cols:
    #         cols[pos[i, 1]] = {'row': [], 'val': []}
    #     rows[pos[i, 0]]['col'].append(pos[i, 1])
    #     rows[pos[i, 0]]['val'].append(val[i])
    #     cols[pos[i, 1]]['row'].append(pos[i, 0])
    #     cols[pos[i, 1]]['val'].append(val[i])
    # for i in range(len(rhos)):
    #     dists = rows[pos[i, 0]]['col'] - pos[i, 1]
    #     vals = rows[pos[i, 0]]['val']
    #     v = np.dot(np.exp(-(dists / 200) ** 2), vals)
    #
    #     dists = cols[pos[i, 1]]['row'] - pos[i, 0]
    #     vals = cols[pos[i, 1]]['val']
    #     v += np.dot(np.exp(-(dists / 200) ** 2), vals)
    #     rhos[i] += 0.5*v

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
    print(len(centroid))

    ########################################################################
    ########################################################################
    # candidates = {'ro': rhos, 'delta': deltas}
    # candidates=pd.DataFrame.from_dict(candidates)
    # candidates['eta'] = candidates['ro'] * candidates['delta']
    # candidates['rank'] = candidates['eta'].rank(ascending=False, method='dense')
    #
    # temp_rank = candidates['rank'] / max(candidates['rank'])
    #
    # temp_eta = candidates['eta'] / max(candidates['eta'])
    #
    # candidates['transformed_rank'] = (temp_rank - temp_eta) / np.sqrt(2)
    # candidates['transformed_eta'] = (temp_eta + temp_rank) / np.sqrt(2)
    # plt.figure()
    # plt.plot(candidates['transformed_rank'], candidates['transformed_eta'], '.')
    # plt.show()
    # # print(candidates.shape, ':canidates shape')
    # # print(candidates['transformed_eta'].idxmin(), 'idxmin of transformed eta')
    # breakpoint = candidates.iloc[candidates['transformed_eta'].idxmin()]['eta']
    # # print(breakpoint, ':breakpoint')
    # candidates['eta_cluster'] = -1
    # candidates.loc[candidates['eta'] > breakpoint, 'eta_cluster'] = candidates.loc[
    #     candidates['eta'] > breakpoint, 'eta_cluster'].rank(ascending=False, method='first')
    #
    # print(candidates)
    # candidates.to_csv('candidates.tsv', sep='\t', header=True, index=False)
    # # while candidates['eta_cluster'].to_list() != previous:
    # #     # print('iteration')
    # #     previous = candidates['eta_cluster'].tolist()
    # #     candidates['eta_cluster'] = candidates.apply(get_nearest_higher_density_cluster, axis=1, candidates=candidates,
    # #                                                  breakpoint=breakpoint)
    # #
    # # candidates = candidates.groupby('eta_cluster').apply(compute_cluster_stats)
    # # candidates = candidates.groupby('eta_cluster').apply(find_cluster_summits, summit_gap=summit_gap)
    ###########################################################################
    ############################################################################



    # deltas[deltas<6]=-1
    # # deltas[deltas > 5] = 1
    theta = rhos * deltas#/(np.max(rhos)*np.max(deltas))
    # theta /=np.max(theta)
    data['rhos']=rhos
    data['deltas']=deltas
    data['theta']=theta
    data['type']= output
    data.to_csv(output, sep='\t', header=False, index=False)

    #
    # print(data)
    # print(data.shape,len(theta),len(deltas),len(rhos))
    import sys
    sys.exit(0)
    # theta[deltas<6]=0
    # theta /= np.max(theta)
    # theta*=2
    thetaorder = np.argsort(-theta)
    n = np.linspace(1, len(theta), len(theta))

    # n /= np.max(n)
    plt.figure()
    plt.plot(n,theta[thetaorder], '.')
    plt.plot([0,1],[0,1])
    plt.show()

    centroid = thetaorder[np.argwhere(theta[thetaorder] > 1000)].flatten()
    # centroid = thetaorder[np.argwhere(theta[thetaorder] / n > 0.5)].flatten()
    # centroid = thetaorder[:700].flatten()
    print('....',len(centroid))

    # assign rest loci to loop clusters
    LargerNei = LargerNei.astype(int)
    label = LargerNei * 0 - 1
    for i in range(len(centroid)):
        label[centroid[i]] = i
    decreasingsortedIdxRhos = np.argsort(-rhos)
    for i in decreasingsortedIdxRhos:
        if label[i] == -1:
            label[i] = label[LargerNei[i]]

    # find noisy loci by asigning cluster halo
    halo = label * 0 - 1
    numLoops = len(set(label))
    broad_rho = np.zeros(numLoops)

    for i in range(len(NNindexes)):
        for j in NNindexes[i]:
            if label[i] != label[j]:
                rho_avg = (rhos[i] + rhos[j]) / 2
                if rho_avg > broad_rho[label[i]]:
                    broad_rho[label[i]] = rho_avg
                if rho_avg > broad_rho[label[j]]:
                    broad_rho[label[j]] = rho_avg

    for i in range(len(halo)):
        if rhos[i] < broad_rho[label[i]]:
            halo[i] = label[i]

    # refine loop
    val = data[6].to_numpy()
    refinedLoop = []
    label = label.flatten()
    for l in set(label):
        idx = np.argwhere((label == l) & (halo == -1)).flatten()
        idx = np.argwhere(label == l).flatten()
        if len(idx) > 0:
            refinedLoop.append(idx[np.argmax(val[idx])])
    val = data[6].to_numpy()  # *data[7].to_numpy()


    # plt.figure()
    # coo = coo_matrix((val, (pos[:, 0].astype(int), pos[:, 1].astype(int))), dtype=float)
    # plt.imshow(coo.toarray())
    # plt.scatter(pos[centroid][:, 1], pos[centroid][:, 0], label='propose')
    # plt.scatter(pos[refinedLoop][:, 1], pos[refinedLoop][:, 0], label='refine propose')
    # target = pd.read_csv('target.bedpe', header=None, sep='\t')
    # target=target[target[0]==chrom]
    # tpos = (target[[1, 4]] // 5000).drop_duplicates().to_numpy()
    # plt.scatter(tpos[:, 1], tpos[:, 0], label='Target', facecolors='black', edgecolors='none')

    # correct = np.array([x for x in set(tuple(x) for x in tpos) & set(tuple(x) for x in pos[refinedLoop])])
    # plt.scatter(correct[:, 1], correct[:, 0], label='match',color='red')
    # plt.title(str(prob)[:10]+','+str(len(correct)))
    # print('refine',candidates,str(len(correct)), ' .....')
    # correct = np.array([x for x in set(tuple(x) for x in tpos) & set(tuple(x) for x in pos[centroid])])
    # print('original',candidates, str(len(correct)), ' .....')
    # plt.legend()
    # plt.xlim([2400, 3400])
    # plt.ylim([2400, 3400])
    # plt.gca().invert_yaxis()
    # plt.show()
    if refine:
        print('find ',len(refinedLoop),'loops..\n save loop to ',output)
        data.loc[refinedLoop].to_csv(output,sep='\t',header=False, index=False)
    else:
        print('find ',len(centroid),'loops..\n save loop to ',output)
        data.loc[centroid].to_csv(output,sep='\t',header=False, index=False)

if __name__ == '__main__':
    pool()
