import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torchlars import LARS

def contrastivePretrain(model,train_dataloader, lr=1e-1, epochs=20, device=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,eps=1e-8)
    optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)
    for epoch in tqdm(range(epochs)):
        totalLoss=0
        model.train()
        for X in train_dataloader:
            loss = 0
            target = X[-1].numpy() # [Batch 1 Class]
            optimizer.zero_grad()
            samples = X[1].shape[1]
            embedding = model(X[1].to(device), X[2].to(device),pretrain=True) # B S H (Bx8x64)
            # print('target.shape',target.shape)
            for tidx in range(target.shape[-1]):
                pos=embedding[np.argwhere(target[...,tidx].flatten()==1).flatten(),:,:]
                # print('pos.shape', pos.shape)
                neg=embedding[np.argwhere(target[...,tidx].flatten()==0).flatten(), :, :]
                # print('neg.shape', neg.shape)
                dim = embedding.shape[-1]
                nPos = pos.shape[0]
                nNeg = neg.shape[0]
                temp=1
                for i in range(nPos):
                    posPair = np.random.choice(samples,2,replace=False)
                    pos_i = pos[i,posPair,:]
                    negIndex = np.random.choice(samples,nNeg,replace=True)
                    batch_neg_i = torch.gather(neg,1,torch.Tensor(np.repeat(negIndex[:,None,None],dim,axis=-1)).to(torch.int64).to(device))
                    batch_neg_i = rearrange(batch_neg_i,'a 1 b -> (b 1) a')
                    pos_and_batch_neg = torch.cat([pos_i[1][:,None],batch_neg_i],dim=-1)
                    sim=model.sim(pos_i[0],pos_and_batch_neg)
                    # sim = torch.matmul(pos_i[0],pos_and_batch_neg) # dot product sim
                    loss-=F.log_softmax(sim/temp,dim=0)[0]

                for i in range(nNeg):
                    negPair = np.random.choice(samples,2,replace=False)
                    neg_i = neg[i,negPair,:]
                    posIndex = np.random.choice(samples,nPos,replace=True)
                    batch_pos_i = torch.gather(pos,1,torch.Tensor(np.repeat(posIndex[:,None,None],dim,axis=-1)).to(torch.int64).to(device))
                    batch_pos_i = rearrange(batch_pos_i,'a 1 b -> (b 1) a')
                    neg_and_batch_pos = torch.cat([neg_i[1][:,None],batch_pos_i],dim=-1)
                    sim=model.sim(neg_i[0],neg_and_batch_pos)
                    loss-=F.log_softmax(sim/temp,dim=0)[0]



            loss.backward()
            optimizer.step()
            totalLoss+=loss
        print('epoch #',epoch, 'loss=',totalLoss)


    return model




