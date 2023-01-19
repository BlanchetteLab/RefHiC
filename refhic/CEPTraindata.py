import pandas as pd


import click
import pickle
from refhic.config import checkConfig
import sys
from refhic.config import referenceMeta
import cooler
import pickle as pkl
import numpy as np
import click


@click.command()
@click.option('--resol', default=5000, help='resolution [5000]')
@click.option('--reference', type=str, default=None, help='a file contains reference panel')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=200, help="submatrix width")
@click.option('--step',type=int,default=None,help='step for submatrix; default: step=w')
@click.option('--chrom',type=str,default=None,help='chromosomes; [All]')
@click.argument('study', type=str,required=True)
@click.argument('target', type=str,required=True)
@click.argument('output', type=str,required=True)
def CEPTraindata(output, resol,study, reference, max_distance,w,target,step,chrom):
    """Create train data for contact map enhancement

    \b
    study: comma separated mcool or bcool files (should be downsampled files) for the study sample
    target: mcool or bcool file as a target
    output: output folder
    \b
    """
    if checkConfig():
        pass
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()


    reference = referenceMeta(reference)


    studyCools  = [cooler.Cooler(file_path+'::/resolutions/'+str(resol)) for file_path in study.split(',')]
    targetCool = cooler.Cooler(target+'::/resolutions/'+str(resol))
    referencedCools = [cooler.Cooler(file_path+'::/resolutions/'+str(resol)) for file_path in reference['file'].to_list()]

    if step is None:
        step = w
    max_distance_bin=max_distance//resol
    if chrom is None:
        chrom = studyCools[0].chromnames
    else:
        chrom = chrom.split(',')
    # remove chrMT,chrY
    for rmchr in ['chrMT','MT','chrM','M','Y','chrY']:
        if rmchr in chrom:
            chrom.remove(rmchr)

    for cchrom in chrom:
        targetMat=targetCool.matrix(balance=True,sparse=True).fetch(cchrom).tocsr()
        targetMat.data[np.isnan(targetMat.data)]=0
        maxval=np.percentile(targetMat.data,99.9)
        targetMat.data[targetMat.data>maxval] = maxval
        targetMat.data/=maxval

        n = targetMat.shape[0]
        studyMats = [studyCool.matrix(balance=True,sparse=True).fetch(cchrom).tocsr() for studyCool in studyCools]
        for i in range(len(studyMats)):
            studyMats[i].data[np.isnan(studyMats[i].data)]=0
            maxval = np.percentile(studyMats[i].data, 99.9)
            studyMats[i].data[studyMats[i].data > maxval] = maxval
            studyMats[i].data /= maxval

        referencedMats = [referencedCool.matrix(balance=True,sparse=True).fetch(cchrom).tocsr() for referencedCool in referencedCools]
        for i in range(len(referencedMats)):
            referencedMats[i].data[np.isnan(referencedMats[i].data)]=0
            maxval = np.percentile(referencedMats[i].data, 99.9)
            referencedMats[i].data[referencedMats[i].data > maxval] = maxval
            referencedMats[i].data /= maxval
        for i in range(0,n-w,step):
            for j in range(0,max_distance_bin,step):
                jj = j+i
                if jj+w<n and i+w<n:
                    # target = [1, 1, w, w] ->  [B, 1, 1, w, w]
                    # referenced = [n, 1, w, w] -> [B, n, 1, w, w]
                    # tests = [N, 1, w, w] -> [1, 1, w, w] -> [B, 1, 1, w, w]
                    _targetMat = targetMat[i:i+w,jj:jj+w].toarray()[None,None,...] # 1, 1, w ,w
                    _studyMats = [studyMat[i:i+w,jj:jj+w].toarray()[None,None,...] for studyMat in studyMats]
                    _referencedMats = [referencedMat[i:i + w, jj:jj + w].toarray()[None, None, ...] for referencedMat in referencedMats]
                    _studyMats=np.vstack(_studyMats)
                    _referencedMats = np.vstack(_referencedMats)


                    with open(output + '/tl_' + cchrom + '_' + str(i * resol) + '_' + str(jj * resol) + '_' + str(
                            resol) + '.pkl', 'wb') as f:
                        pkl.dump( [_studyMats,_referencedMats,_targetMat], f)







if __name__ == '__main__':
    CEPTraindata()





