import torch.nn as nn
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.nn.modules import loss

class encoder(nn.Module):
    '''
    Model used for detect/locate loops from a (sub)-contact map, with (additional features, i.e. prob from the output of grouploop)
    '''
    def __init__(self,input_size, in_channels,encoding_dim,win):
        super(encoder, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.win=win


        self.FC = nn.Sequential(
            nn.BatchNorm1d(self.input_size-self.win*self.win*0),
            nn.Linear(self.input_size-self.win*self.win*0, self.encoding_dim),
            nn.BatchNorm1d(self.encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.BatchNorm1d(self.encoding_dim),
            nn.ReLU()
        )
        self.BN=nn.BatchNorm1d(input_size)

        self.CNN = nn.Sequential(
            nn.InstanceNorm2d(self.in_channels, affine=True),
            nn.Conv2d(self.in_channels,self.in_channels,kernel_size=3,padding='same'),
            nn.ReLU(),
        )

    def forward(self, x):
        batchsize=x.shape[0]
        x=rearrange(x,'a b c -> (a b) c')
        x=self.BN(x)
        x2D=rearrange(x[:,:self.win*self.win*2],'a (c d e) -> a c d e',c=2,d=self.win)
        x1D=x[:,self.win*self.win*2:]

        # cnn1out=self.CNN1(x2D)
        # cnn2out=self.CNN2(x2D)
        # cnnout = torch.cat([x2D,cnn1out,cnn2out],1)
        # cnnout = self.CNNproject(cnnout)

        cnnout=self.CNN(x2D)
        cnnout=torch.flatten(cnnout,1)
        y=torch.cat([cnnout,x1D],dim=-1)
        y = self.FC(y)
        y = rearrange(y,'(a b) c -> a b c',a=batchsize)
        return y


class ensembles(nn.Module):
    def __init__(self,models=None):
        super().__init__()
        self.models=[]
        for m in models:
            self.models.append(m)
    def forward(self,x,xs):
        prob=0
        for m in self.models:
            prob+=m(x,xs)
        return prob/len(self.models)

class refhicNet(nn.Module):
    def __init__(self, input_size, encoding_dim=128,header=8,CNNencoder=True,win=21,classes=1):
        super(refhicNet, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.header=header
        self.CNNencoder=CNNencoder
        self.win=win
        self.classes=classes

        if CNNencoder:
            self.encoder = encoder(self.input_size,2,self.encoding_dim,self.win)
        else:
            self.encoder = nn.Sequential(
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, encoding_dim),
                nn.BatchNorm1d(self.encoding_dim),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(encoding_dim, self.encoding_dim),
                nn.BatchNorm1d(self.encoding_dim),
                nn.ReLU(),
            )


        self.attentionFC = nn.Sequential(
            nn.LayerNorm(self.encoding_dim),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.ReLU(),
        )

        self.layernorm = nn.LayerNorm(self.encoding_dim)

        self.predictor = nn.Sequential(
            nn.BatchNorm1d(self.encoding_dim*2),
            nn.Linear(self.encoding_dim*2 , self.encoding_dim*2),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.encoding_dim*2, self.encoding_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.encoding_dim , self.classes),
        )

        self.dummy_para=nn.Parameter(torch.empty(0))

    def pretrain(self, x):
        batchSize = x.shape[0]
        sampleSize = x.shape[1]

        if self.CNNencoder:
            encoding=self.encoder(x)
        else:
            x = rearrange(x, 'a b c -> (a b) c')
            encoding=self.encoder(x).view(batchSize, sampleSize,self.encoding_dim)
        encoding = self.layernorm(encoding)
        return encoding

    def sim(self,x,y):
        sim=torch.matmul(x,y)
        return sim

    def forward(self, x0,x1,returnAtten=False,pretrain=False):
        x = [x0,x1]
        x[0] = x[0].float()
        x[1] = x[1].float()
        if pretrain:
            return self.pretrain(x[0])

        batchSize = x[0].shape[0]

        if self.CNNencoder:
            input = torch.cat([x[0][:, None, :], x[1]], 1)

            allEncoding = self.encoder(input)
        else:
            input = rearrange(torch.cat([x[0][:, None, :], x[1]], 1), 'a b c -> (a b) c')
            allEncoding = rearrange(self.encoder(input), '(a b) c -> a b c',a=batchSize)

        testEncoding = allEncoding[:,0,:]
        x[1] = allEncoding
        x[1] = self.layernorm(x[1])
        x[0] = x[1][:,[0],:]
        x[1] = x[1][:,1:,:]




        Q = x[0]
        V = x[1]
        K = rearrange(x[1], 'a b c -> a c b')


        sim = torch.matmul(Q,K)
        alpha = torch.softmax(sim / (self.encoding_dim ** 0.5), dim=-1)
        attention = torch.matmul(alpha, V).squeeze(-2)

        x = attention+self.attentionFC(attention)


        output = self.predictor(torch.cat((x, testEncoding), -1))

        if returnAtten:
            return output,alpha
        return output




class baselineNet(nn.Module):
    def __init__(self, input_size, encoding_dim=128,CNNencoder=True,win=21):
        super().__init__()
        self.input_size=input_size
        self.encoding_dim=encoding_dim
        self.CNNencoder=CNNencoder
        self.win = win

        if CNNencoder:
            self.encoder = encoder(self.input_size,2,self.encoding_dim,self.win)
        else:
            self.encoder = nn.Sequential(
                nn.BatchNorm1d(self.input_size),
                nn.Linear(self.input_size, encoding_dim),
                nn.BatchNorm1d(self.encoding_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(encoding_dim, self.encoding_dim),
                nn.BatchNorm1d(self.encoding_dim),
                nn.ReLU(),
            )

        self.MLP = nn.Sequential(
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.BatchNorm1d(self.encoding_dim),
            nn.ReLU(),
            nn.Linear(self.encoding_dim , self.encoding_dim // 2),
            nn.BatchNorm1d(self.encoding_dim//2),
            nn.ReLU(),
            nn.Linear(self.encoding_dim//2, self.encoding_dim // 2),
            nn.BatchNorm1d(self.encoding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.encoding_dim // 2, 1),
        )

    def forward(self, x):
        x = x.float()
        if self.CNNencoder:
            x=x[:,None,:]
        x = self.encoder(x)
        if self.CNNencoder:
            x=x[:,0,:]
        output = self.MLP(x)
        return output


class focalLoss(loss._WeightedLoss):
    __constants__ = ['reduction']
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "mean",adaptive=False):
        super(focalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.adaptive=adaptive
        print('......using adaptive gamma,',self.adaptive)
    def get_gamma(self,p_t):
        p_t = p_t.detach()
        gammas=p_t*0
        gammas[p_t<0.5] = 3
        gammas[p_t < 0.2] = 5
        gammas[p_t>=0.5] = self.gamma
        return gammas
    def forward(self,inputs,targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        if self.adaptive:
            gamma=self.get_gamma(p_t)
        else:
            gamma=self.gamma

        loss = ce_loss * ((1 - p_t) ** gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss