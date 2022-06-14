import numpy as np

def fdr(target,decoy,alpha=0.05):
    '''
    perform FDR control to select true positive target samples at a user specific alpha. alpha=0.05 means 100 samples containing 5 decoy datapoints
    :param target: list of float scores from target dataset; the higher score, the better
    :param decoy: list of float scores from decoy dataset
    :param alpha: FDR alpha level, default 0.05
    :return:
        cutoff: minimum true positive scores
    '''
    eps = np.finfo(float).eps
    target = np.asarray(target)
    targetLabel = target * 0 + True
    decoy = np.asarray(decoy)
    decoyLabel = decoy * 0 + False
    val = np.concatenate([target, decoy])
    label = np.concatenate([targetLabel, decoyLabel])
    reverseArgSort = np.argsort(val)[::-1]
    val = val[reverseArgSort]
    label = label[reverseArgSort]

    numOfTarget = 0
    numOfDecoy = 0
    while numOfTarget + numOfDecoy == 0 or numOfDecoy / (numOfTarget+eps) <= alpha:
        idx = numOfTarget + numOfDecoy
        cutoff = val[idx]
        if label[idx]:
            numOfTarget += 1
        else:
            numOfDecoy += 1
    if numOfDecoy / (numOfTarget+eps) > alpha:
        cutoff -= eps
    return cutoff
