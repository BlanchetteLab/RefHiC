from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal
import torchmetrics
import tqdm

# Loss functions
def loss_coteaching(pred1, pred2, target, criterion,forget_rate,reduction='mean'):
    pred1 = pred1.view(-1)
    pred2 = pred2.view(-1)
    target = target.view(-1)
    loss1 = criterion(pred1, target)
    idx1_sorted = torch.argsort(loss1.data)



    loss2 = criterion(pred2, target)
    idx2_sorted = torch.argsort(loss2.data)


    remember_rate = 1 - forget_rate


    num_remember = int(remember_rate * len(idx1_sorted))


    if num_remember == 0:
        num_remember = len(idx1_sorted)

    idx1_update = idx1_sorted[:num_remember]
    idx2_update = idx2_sorted[:num_remember]


    loss1_update = torch.mean(criterion(pred1[idx2_update], target[idx2_update]))
    loss2_update = torch.mean(criterion(pred2[idx1_update], target[idx1_update]))

    return loss1_update, loss2_update


def loss_coteaching_plus(pred1, pred2, target,criterion, forget_rate,reduction='mean'):
    pred1 = pred1.view(-1)
    pred2 = pred2.view(-1)
    target = target.view(-1)
    logical_disagree_id = (pred1>0.5)!=(pred2>0.5)

    disagree_id = torch.nonzero(logical_disagree_id).view(-1)
    # print('disagree_id',disagree_id,len(disagree_id))

    if len(disagree_id) > 0:

        update_target = target[disagree_id]
        update_pred1 = pred1[disagree_id]
        update_pred2 = pred2[disagree_id]
        print(len(update_pred2))
        loss1, loss2 = loss_coteaching(update_pred1, update_pred2, update_target,criterion,forget_rate)
    else:
        print('two network produce the same result')
        loss1 = torch.mean(criterion(pred1, target))
        loss2 = torch.mean(criterion(pred2, target))

    return loss1, loss2

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def gen_forget_rate(maxEpoch=100,numGradual=10,forgetRate=0.5):
    rate_schedule = np.ones(maxEpoch)*forgetRate
    rate_schedule[:numGradual] = np.linspace(0, forgetRate, numGradual)
    return rate_schedule


def coteachingPlusTrain(model1,model2, optimizer1, optimizer2, train_dataloader,criterion, epoch, rateSchedule,device,batchsize=128,TBWriter=None,baseline=False,init_epoch=5):


    model1.train()
    model2.train()
    preds1 = []
    preds2 = []
    targets = []
    # print('training epoch',epoch)
    for batch_idx,X in enumerate(train_dataloader):
        target = X[-1].to(device)
        pred1 = model1(X[0].to(device),X[1].to(device))
        pred2 = model2(X[0].to(device),X[1].to(device))
        preds1.append(pred1.cpu())
        targets.append(target.cpu())
        preds2.append(pred2.cpu())

        if epoch < init_epoch:
            loss1, loss2= loss_coteaching(pred1, pred2, target, criterion,rateSchedule[epoch])
        else:
            loss1, loss2= loss_coteaching_plus(pred1, pred2, target, criterion, rateSchedule[epoch])
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    preds1=torch.vstack(preds1).flatten()
    preds2 = torch.vstack(preds2).flatten()
    targets=torch.vstack(targets).flatten()
    loss1 = torch.mean(criterion(preds1, targets))
    loss2 = torch.mean(criterion(preds2, targets))
    acc1 = torchmetrics.functional.accuracy(preds1, targets.to(int))
    f11 = torchmetrics.functional.f1(preds1, targets.to(int))
    precision1, recall1 = torchmetrics.functional.precision_recall(preds1, targets.to(int))

    acc2 = torchmetrics.functional.accuracy(preds2, targets.to(int))
    f12 = torchmetrics.functional.f1(preds2, targets.to(int))
    precision2, recall2 = torchmetrics.functional.precision_recall(preds2, targets.to(int))

    if TBWriter:
        TBWriter.add_scalar("Loss/train1", loss1, epoch)
        TBWriter.add_scalar("Accuracy/train1", acc1, epoch)
        TBWriter.add_scalar("F1/train1", f11, epoch)
        TBWriter.add_scalar("Precision/train1", precision1, epoch)
        TBWriter.add_scalar("Recall/train1", recall1, epoch)
        TBWriter.add_scalar("Loss/train2", loss2, epoch)
        TBWriter.add_scalar("Accuracy/train2", acc2, epoch)
        TBWriter.add_scalar("F1/train2", f12, epoch)
        TBWriter.add_scalar("Precision/train2", precision2, epoch)
        TBWriter.add_scalar("Recall/train2", recall2, epoch)







