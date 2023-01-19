import torch
from refhic.SRModels import RefHiCSRNet
from tqdm import tqdm
from refhic.config import checkConfig,loadConfig,referenceMeta
import sys
import cooler
import numpy as np
import click


@click.command()
@click.option('--vmin', type=float, default=1e-5, help='enhanced values smaller than vmin are set as 0')
@click.option('--resol', default=5000, help='resolution [5000]')
@click.option('--reference', type=str, default=None, help='a file contains reference panel')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('-w', type=int, default=200, help="submatrix width")
@click.option('--step',type=int,default=None,help='step for submatrix; default: step=w')
@click.option('--chrom',type=str,default=None,help='chromosomes; [All]')
@click.option('--crop',type=int,default=10,help='crop each end by x-bin')
@click.option('--cpu', type=bool, default=False, help='Use CPU [False]')
@click.option('--gpu', type=int, default=None, help='GPU index [auto select]')
@click.option('--modelState',type=str,default =None,help='trained model')
@click.argument('study', type=str,required=True)
# @click.argument('target', type=str,required=True)
@click.argument('output', type=str,required=True)
def pred(output, vmin,resol,study, reference, max_distance,w,step,chrom,crop,cpu,gpu,modelstate):
    '''Enhancing Hi-C contact maps'''
    if checkConfig():
        config=loadConfig()
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()

    reference = referenceMeta(reference)

    if cpu:
        device = torch.device("cpu")
        print('use CPU ...')
    else:
        if torch.cuda.is_available():
            if gpu is not None:
                device = torch.device("cuda:"+str(gpu))
                print('use gpu '+ "cuda:"+str(gpu))
            else:
                gpuIdx = torch.cuda.current_device()
                device = torch.device(gpuIdx)
                print('use gpu ' + "cuda:" + str(gpuIdx))
        else:
            device = torch.device("cpu")
            print('GPU is not available, use CPU ...')

    if modelstate is None:
        modelstate=config['sr']['model']
    output = open(output, 'w')
    # modelstate = torch.load(modelstate,map_location=device)#['parameters']
    # # print(parameters)
    # print('***************************')
    # if 'model' not in parameters:
    #     print('Model: unknown')
    # else:
    #     print('Model:',parameters['model'])
    # print('Resolution:',parameters['resol'])
    # print('window size:',parameters['w']*2+1)
    # print('***************************')
    model = RefHiCSRNet(w=200).to(device)
    _modelstate = torch.load(modelstate, map_location=device)
    model.load_state_dict(_modelstate,strict=True)
    model.eval()


    studyCool = cooler.Cooler(study+'::/resolutions/'+str(resol))
    # targetCool = cooler.Cooler(target + '::/resolutions/' + str(resol))
    referencedCools = [cooler.Cooler(file_path+'::/resolutions/'+str(resol)) for file_path in reference['file'].to_list()]

    if step is None:
        step = w

    step-=2*crop
    max_distance_bin=max_distance//resol
    if chrom is None:
        chrom = studyCool.chromnames
    else:
        chrom = chrom.split(',')
    # remove chrMT,chrY
    for rmchr in ['chrMT','MT','chrM','M','Y','chrY']:
        if rmchr in chrom:
            chrom.remove(rmchr)
    with torch.no_grad():
        for cchrom in tqdm(chrom):
            studyMat=studyCool.matrix(balance=True,sparse=True).fetch(cchrom).tocoo()#tocsr()
            studyMat.data[studyMat.col - studyMat.row > max_distance_bin] = 0
            studyMat = studyMat.tocsr()
            studyMat.data[np.isnan(studyMat.data)]=0
            maxval=np.percentile(studyMat.data,99.9)
            studyMat.data[studyMat.data>maxval] = maxval
            studyMat.data/=maxval
            n = studyMat.shape[0]

            # targetMat=targetCool.matrix(balance=True,sparse=True).fetch(cchrom).tocoo()
            # targetMat.data[targetMat.col - targetMat.row > max_distance_bin] = 0
            # targetMat = targetMat.tocsr()
            #
            # targetMat.data[np.isnan(targetMat.data)]=0
            # maxval=np.percentile(targetMat.data,99.9)
            # targetMat.data[targetMat.data>maxval] = maxval
            # targetMat.data/=maxval


            referencedMats = [referencedCool.matrix(balance=True,sparse=True).fetch(cchrom).tocsr() for referencedCool in referencedCools]
            # referencedMats = [studyCool.matrix(balance=True,sparse=True).fetch(cchrom).tocsr() for i in range(30)]
            for i in range(len(referencedMats)):
                referencedMats[i].data[np.isnan(referencedMats[i].data)]=0
                maxval = np.percentile(referencedMats[i].data, 99.9)
                referencedMats[i].data[referencedMats[i].data > maxval] = maxval
                referencedMats[i].data /= maxval
            for i in range(0,n-w,step):
                for j in range(0,max_distance_bin+step,step):
                    jj = j+i
                    if jj+w<n and i+w<n:
                        # study = [1, 1, w, w] ->  [B, 1, 1, w, w]
                        # referenced = [n, 1, w, w] -> [B, n, 1, w, w]
                        _studyMat = studyMat[i:i+w,jj:jj+w].toarray()[None,None,...] # 1, 1, w ,w
                        # _targetMat = targetMat[i:i + w, jj:jj + w].toarray()[None, None, ...]  # 1, 1, w ,w
                        _referencedMats = [referencedMat[i:i + w, jj:jj + w].toarray()[None, None, ...] for referencedMat in referencedMats]
                        _referencedMats = np.vstack(_referencedMats)
                        # print(_studyMat.shape,_referencedMats.shape)
                        X=torch.Tensor(np.concatenate((_studyMat[None,...], _referencedMats[None,...]), 1)).to(device)
                        # print(X.shape,'............')
                        # break
                        X = model(X).cpu().detach().numpy()
                        # print(X.shape)
                        CoorTop = i+crop
                        CoorLeft = jj+crop

                        X=X[...,crop:-crop,crop:-crop]
                        lowRes = _studyMat[...,crop:-crop,crop:-crop]
                        # highRes = _targetMat[...,crop:-crop,crop:-crop]
                        X[X>1] = 1
                        X[X<vmin] = 0
                        nne = lowRes + X  # help to get nnz in both input and output
                        idx = np.nonzero(nne[0, 0, ...])
                        val=X[0, 0, idx[0], idx[1]]
                        lowRes = lowRes[0, 0, idx[0], idx[1]]
                        for _i in range(len(val)):
                            bin1 = idx[0][_i]+CoorTop
                            bin2 = idx[1][_i]+CoorLeft
                            if bin2>=bin1 and bin2-bin1<=max_distance_bin:
                                output.write('\t'.join([str(field) for field in
                                                        [cchrom, bin1 * resol,bin1 * resol+resol,cchrom, bin2 * resol,bin2 * resol+resol, val[_i], lowRes[_i]
                                                         ]
                                                        ]) + '\n')






if __name__ == '__main__':
    pred()
