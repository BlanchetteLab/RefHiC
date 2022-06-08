import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5py
from groupLoopModels import locateNet,focalLoss, CBLoss
import numpy as np
from gcooler import gcool
import click
from data import  matricesPatchDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import scipy
from util import rangeSplit

def trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize=128,TBWriter=None):
    model.train()
    preds=[]
    targets = []
    for batch_idx,X in enumerate(train_dataloader):
        target = X[-1].to(device)
        input = X[0].to(device)
        # [128, 2, 100, 100]
        mask=input[:,1,:,:]>0.5
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[:, 0, :,:][mask],target[mask])
        preds.append(output[:, 0, :,:][mask].cpu())
        targets.append(target[mask].cpu())
        # loss = criterion(output.flatten(), target.flatten())
        # preds.append(output.flatten().cpu())
        # targets.append(target.flatten().cpu())
        loss.backward()
        optimizer.step()
    preds=torch.cat(preds)
    targets=torch.cat(targets)
    loss = criterion(preds, targets)
    acc = torchmetrics.functional.accuracy(preds, targets.to(int))
    f1 = torchmetrics.functional.f1(preds, targets.to(int))
    precision, recall = torchmetrics.functional.precision_recall(preds, targets.to(int))
    positive = torch.sum(preds >= 0.5)
    negative = torch.sum(preds < 0.5)
    # allPos = torch.sum(targets==1)
    # allNeg = torch.sum(targets == 0)
    if TBWriter:
        TBWriter.add_scalar("Loss/train", loss, epoch)
        TBWriter.add_scalar("Accuracy/train", acc, epoch)
        TBWriter.add_scalar("F1/train", f1, epoch)
        TBWriter.add_scalar("Precision/train", precision, epoch)
        TBWriter.add_scalar("Recall/train", recall, epoch)
        TBWriter.add_scalar("positive/train", positive, epoch)
        TBWriter.add_scalar("negative/train", negative, epoch)



def testModel(model, test_dataloader,criterion, device,epoch,printData=False,TBWriter=None):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    preds = []
    with torch.no_grad():
        for X in (test_dataloader):
            target = X[-1].to(device)
            input = X[0].to(device)
            output = model(input)
            preds.append(output.flatten().cpu())
            targets.append(target.flatten().cpu())




    preds=torch.vstack(preds).flatten()
    targets=torch.vstack(targets).flatten()
    loss = criterion(preds, targets)
    acc = torchmetrics.functional.accuracy(preds, targets.to(int))
    f1 = torchmetrics.functional.f1(preds, targets.to(int))
    precision, recall = torchmetrics.functional.precision_recall(preds, targets.to(int))
    positive = torch.sum(preds >= 0.5)
    negative = torch.sum(preds < 0.5)
    allPos = torch.sum(targets==1)
    allNeg = torch.sum(targets == 0)
    if TBWriter:
        TBWriter.add_scalar("Loss/test", loss, epoch)
        TBWriter.add_scalar("Accuracy/test", acc, epoch)
        TBWriter.add_scalar("F1/test", f1, epoch)
        TBWriter.add_scalar("Precision/test", precision, epoch)
        TBWriter.add_scalar("Recall/test", recall, epoch)
        TBWriter.add_scalar("positive/test", positive, epoch)
        TBWriter.add_scalar("negative/test", negative, epoch)
        TBWriter.add_scalar("allPos/test", allPos, epoch)
        TBWriter.add_scalar("allNeg/test", allNeg, epoch)

def getTrainTestPatches(nBins,maxDistanceBins,size=100,overlap=10,test_size=0.1):
    rowSegs = rangeSplit(0,nBins,size,overlap)
    if nBins-size//2>rowSegs[-1][-1]:
        rowSegs.append((nBins-size,nBins))
    colSegs = []
    for startRow,endRow in rowSegs:
        startCol = startRow
        endCol = np.min([endRow+maxDistanceBins,nBins])
        _colSegs = rangeSplit(startCol,endCol,size,overlap)
        if len(_colSegs)>0 and endCol - size // 2 > rowSegs[-1][-1]:
            _colSegs.append((endCol - size, endCol))
        if len(_colSegs) > 0:
            colSegs.append(_colSegs)
    patches = []
    for i in range(len(rowSegs)):
        top,bottom = rowSegs[i]
        for j in range(len(colSegs[i])):
            left,right = colSegs[i][j]
            patches.append([top,bottom,left,right])
    patches_train, patches_test = train_test_split(patches, test_size=test_size,random_state=42, shuffle=True)
    return patches_train, patches_test

@click.command()
@click.option('--lr', type=float, default=1e-3, help='learning rate')
@click.option('--name',type=str,default='', help ='training name')
@click.option('--batchsize', type=int, default=512, help='batch size')
@click.option('--epochs', type=int, default=20, help='training epochs')
@click.option('--gpu', type=bool, default=True, help='GPU training')
@click.option('--chrom', type=str, default=None, help='chromosome training')
@click.option('--resol', default=5000, help='resolution')
@click.option('--prob',type=str,default=None, help = '.bedpe file containing groupLoop`s prob output')
@click.option('--target',type=str,default=None, help = '.bedpe file containing targets')
@click.option('--test', type=str, default=None, help='test .gcool')
@click.option('--max_distance', type=int, default=3000000, help='max distance (bp) between contact pairs')
@click.option('--focal_alpha', type=float, default=0.25, help='focal loss alpha')
@click.option('--focal_gamma', type=float, default=2, help='focal loss alpha')
# @click.option('--models',type=str,default ='groupLoop',help='groupLoop; baseline; groupLoop,baseline')
def trainAttention(lr,name,batchsize, epochs, gpu, chrom, resol, test,prob,target, max_distance,focal_alpha,focal_gamma):
    if gpu:
        device = torch.device("cuda")
        print('use gpu')
    else:
        device = torch.device("cpu")

    g = gcool(test+'::/resolutions/'+str(resol))
    if 'chr' not in chrom:
        chrom = 'chr'+chrom
    cmap = g.matrix(balance=True, sparse=True).fetch(chrom).tocsc()
    probf = pd.read_csv(prob,sep='\t',header=None)
    probf = probf[probf[0] == chrom]
    pmap = scipy.sparse.coo_matrix((probf[6], (probf[1] // resol, probf[4] // resol)), shape=cmap.shape).tocsc()
    targetf = pd.read_csv(target,sep='\t',header=None)
    targetf = targetf[targetf[0] == chrom]
    tmap = scipy.sparse.coo_matrix((targetf[6], (targetf[1] // resol, targetf[4] // resol)), shape=cmap.shape).tocsc()
    nBins = cmap.shape[0]
    overlap = 10
    size = 100
    patchesTrain,patchesTest = getTrainTestPatches(nBins,max_distance//resol,size,overlap,test_size=0.1)
    training_data = matricesPatchDataset([cmap,pmap],patchesTrain,tmap)
    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True,num_workers=1)
    test_data = matricesPatchDataset([cmap,pmap],patchesTest,tmap)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=True,num_workers=1)

    print('#train',len(patchesTrain),'\n#test',len(patchesTest))
    model = locateNet(in_channels = 2).to(device)
    TBWriter = SummaryWriter(comment=' LocateNet '+name)
    model.train()
    # criterion = focalLoss(alpha=focal_alpha,gamma=focal_gamma,reduction='sum')
    criterion = CBLoss(samples_per_cls=[nBins*max_distance//resol,tmap.sum()],gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    for epoch in tqdm(range(1, epochs + 1)):
        trainModel(model,train_dataloader, optimizer, criterion, epoch, device,batchsize,TBWriter=TBWriter)
        testModel(model,test_dataloader, criterion,device,epoch,TBWriter=TBWriter)
        # scheduler.step()

    torch.save(model.state_dict(), 'groupLoopLocate+'+chrom+'_state.h5')
    torch.save(model, 'groupLoopLocate+'+chrom+'_Model.h5')









if __name__ == '__main__':
    trainAttention()