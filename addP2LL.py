import pandas as pd
import numpy as np
from gcooler import gcool
import click
@click.command()
@click.option('--resol',type=int,default=5000,help='resolution')
@click.option('--w',type=int,default=10,help='window')
@click.argument('gcoolfile', type=str,required=True)
@click.argument('bedpe', type=str,required=True)
def addP2LL(resol,gcoolfile,bedpe,w):
    f=pd.read_csv(bedpe,sep='\t',header=None)
    g = gcool(gcoolfile+'::/resolutions/'+str(resol))
    chroms = set(f[0])
    for chrom in chroms:
        pos = f[f[0]==chrom][[1,4]].to_numpy()
        p2ll=[]
        nnz=[]
        bmatrix = g.bchr(chrom)
        for i in range(pos.shape[0]):
            mat, meta = bmatrix.square(pos[i,0], pos[i,1], w, 'oe')
            c=mat.shape[1]//2
            _nnz=np.sum(mat[0,c - 1:c + 2, c - 1:c + 2]>0)
            nnz.append(_nnz)
            p2ll.append(meta[-3])
        f.loc[f[0]==chrom,'p2ll']=p2ll
        f.loc[f[0] == chrom, 'nnz'] = nnz
    f.to_csv('.'.join(bedpe.split('.')[:-1])+'_p2ll_nnz.bedpe',header=False,index=False,sep='\t')

if __name__ == '__main__':
    addP2LL()