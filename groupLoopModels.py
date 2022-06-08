import torch.nn as nn
import torch
from einops import rearrange
from torch.nn import functional as F


class residualBlock(nn.Module):
    def __init__(self, in_channels):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.BN2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.BN1(self.conv1(x))
        y = torch.relu(y)
        y = self.BN2(self.conv2(y))
        y = torch.relu(x + y)
        return y


class residualBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(residualBlocks, self).__init__()
        if (in_channels != out_channels):
            blocks = [nn.Conv2d(in_channels, out_channels, 3, padding='same'), nn.ReLU()]
        else:
            blocks = []
        for i in range(1, num_blocks):
            blocks.append(residualBlock(out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


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
        print('input_size',input_size)

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

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            self.noise=self.noise.to(x.device)
            # print(self.noise)
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x

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

class attentionToAdditionalHiC(nn.Module):
    def __init__(self, input_size, encoding_dim=128,header=8,CNNencoder=True,win=21,classes=1):
        super(attentionToAdditionalHiC, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.header=header
        self.CNNencoder=CNNencoder
        self.win=win
        self.classes=classes
        print('self.win',self.win)
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

        # self.gating = nn.Sequential(
        #     nn.BatchNorm1d(self.input_size*2),
        #     nn.Linear(self.input_size*2,self.input_size),
        #     nn.Dropout(0.25),
        #     nn.ReLU(),
        #     nn.Linear(self.input_size,self.input_size//4),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(self.input_size//4,1),
        #     nn.Sigmoid()
        # )
        # self.att = nn.Parameter(torch.zeros((self.encoding_dim, self.encoding_dim)))
        # nn.init.eye_(self.att) # better

        # self.singledataFC = nn.Sequential(
        #     nn.BatchNorm1d(self.encoding_dim),
        #     nn.Linear(self.encoding_dim, self.encoding_dim),
        #     nn.Dropout(0.25),
        #     nn.ReLU(),
        #     nn.Linear(self.encoding_dim, self.encoding_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(self.encoding_dim, self.encoding_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(self.encoding_dim, self.encoding_dim),
        #     nn.ReLU(),
        # )

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
        # self.posTpl=torch.from_numpy(np.loadtxt('posVal.txt')).float()
        # self.negTpl=torch.from_numpy(np.loadtxt('negVal.txt')).float()
        # self.tpl=torch.cat([self.posTpl[None,None,:],self.negTpl[None,None,:]],dim=1)
        # self.tpl=self.tpl[:,:,:self.input_size]
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

        # btpl=torch.tile(self.tpl,(batchSize,1,1)).to(self.dummy_para.device)

        # print('input.shape',input.shape)
        if self.CNNencoder:
            input = torch.cat([x[0][:, None, :], x[1]], 1)
            #print(x[0][:,None,:].shape,x[1].shape,self.btpl.shape)
            allEncoding = self.encoder(input)
        else:
            input = rearrange(torch.cat([x[0][:, None, :], x[1]], 1), 'a b c -> (a b) c')
            allEncoding = rearrange(self.encoder(input), '(a b) c -> a b c',a=batchSize)
        # print('allEncoding.shape',allEncoding.shape)
        testEncoding = allEncoding[:,0,:]
        x[1] = allEncoding
        x[1] = self.layernorm(x[1])
        x[0] = x[1][:,[0],:]
        x[1] = x[1][:,1:,:]




        Q = x[0]
        V = x[1]
        K = rearrange(x[1], 'a b c -> a c b')
        # # MHA
        # # Q,K,V torch.Size([449, 1, 64]) torch.Size([449, 64, 10]) torch.Size([449, 10, 64])
        # Q = rearrange(Q,"b s (h d) -> b h s d",h=self.header)
        # K = rearrange(K, "b (h d) s -> b h d s", h=self.header)
        # V = rearrange(V, "b s (h d) -> b h s d", h=self.header)
        # # print('Q,K,V', Q.shape, K.shape, V.shape) # Q,K,V torch.Size([456, 8, 1, 8]) torch.Size([456, 8, 8, 10]) torch.Size([456, 8, 10, 8])
        # sim = torch.matmul(Q, K)
        # alpha = torch.softmax(sim / ((self.encoding_dim//self.header) ** 0.5), dim=-1)
        # attention = rearrange(torch.matmul(alpha, V),'b h s d -> b s (h d)').squeeze(-2)
        #
        # # end of MHA

        sim = torch.matmul(Q,K)
        alpha = torch.softmax(sim / (self.encoding_dim ** 0.5), dim=-1)
        attention = torch.matmul(alpha, V).squeeze(-2)
        # x = testEncoding + attention
        # x = x + self.attentionFC(x)
        x = attention+self.attentionFC(attention)


        output = self.predictor(torch.cat((x, testEncoding), -1))
        # output = self.predictor(x)
        if returnAtten:
            return output,alpha
        return output




class baseline(nn.Module):
    def __init__(self, input_size, encoding_dim=128,CNNencoder=True):
        super().__init__()
        self.input_size=input_size
        self.encoding_dim=encoding_dim
        self.CNNencoder=CNNencoder

        if CNNencoder:
            self.encoder = encoder(self.input_size,2,self.encoding_dim)
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





from torch.nn.modules import loss

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
        # targets[targets==1] = 0.9
        # targets[targets==0] = 0.1
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

from losses import CB_loss

class CBLoss(loss._WeightedLoss):
    def __init__(self, samples_per_cls: list, no_of_classes: int =2, beta: float = 0.9999, gamma: float = 2,loss_type='focal'):
        super(CBLoss,self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.samples_per_cls = samples_per_cls
        self.no_of_classes =no_of_classes

    def forward(self,preds,targets):
        return CB_loss(targets.flatten(), preds.flatten(), self.samples_per_cls, self.no_of_classes,self.loss_type, self.beta, self.gamma)


class elrBCE_loss(loss._WeightedLoss):
    def __init__(self, num_examp, Lambda = 3, beta=0.7,device=None):
        """:param num_examp:
        :param num_classes:
        :param beta:
        Early Learning Regularization for BCE.
        Parameters
        `num_examp` Total number of training examples.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super(elrBCE_loss, self).__init__()
        self.target = torch.zeros(num_examp,2)
        if device:
            self.target = self.target.to(device)
        self.beta = beta
        self.Lambda = Lambda

    def forward(self, index, logits, label,train=True):
        """Early Learning Regularization.
        Args
        * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
        * `output` Model's logits, same as PyTorch provided loss functions.
        * `label` Labels, same as PyTorch provided loss functions.
        """
        pred = torch.sigmoid(logits)
        pred = torch.cat([pred[:,None],1-pred[:,None]],axis=1)
        pred = torch.clamp(pred, 1e-4, 1.0-1e-4)
        pred_ = pred.data.detach()
        if train:
            self.target[index] = self.beta * self.target[index] + (1-self.beta) * pred_/(pred_).sum(dim=1,keepdim=True)
        bce_loss = F.binary_cross_entropy_with_logits(logits, label)

        elr_reg = ((1-(self.target[index] * pred).sum(dim=1)).log()).mean()


        final_loss = bce_loss + self.Lambda *elr_reg
        return final_loss


import numpy as np
class groupLoopZero(attentionToAdditionalHiC):
    def __init__(self, input_size, encoding_dim=128,testCelltype=None,extraCelltypes=None,batchsize=64,device=None):
        super(groupLoopZero, self).__init__(input_size, encoding_dim)
        self.testCelltype = torch.from_numpy(np.repeat(testCelltype[None],batchsize,axis=0)).to(device)
        _extraCelltypes = rearrange(np.repeat(extraCelltypes[None],batchsize,axis=0),'a b c -> a c b')
        self.extraCelltypes = torch.from_numpy(_extraCelltypes).to(device)
        print('self.extraCelltypes.shape',self.extraCelltypes.shape)
        print('self.testCelltype.shape', self.testCelltype.shape)
        cellTypeDim = self.testCelltype.shape[-1]
        print('cellTypeDim',cellTypeDim)
        self.celltypeAtt = nn.Parameter(torch.zeros((cellTypeDim, cellTypeDim)))
        nn.init.eye_(self.celltypeAtt) # better


    def forward(self, x0,x1):
        x = [x0,x1]
        x[0] = x[0].float()
        x[1] = x[1].float()
        batchSize = x[0].shape[0]
        sampleSize = x[1].shape[1]


        testEncoding = self.encoder(x[0]*0)
        restEncoding = self.encoder(x[1].view(batchSize * sampleSize, self.input_size)).view(batchSize, sampleSize,
                                                                                             self.encoding_dim)

        # K = restEncoding.permute(0, 2, 1)
        K = self.extraCelltypes[:batchSize,...]
        Q = self.testCelltype[:batchSize,...]
        print('K.shape',K.shape)
        QW = torch.matmul(Q, self.celltypeAtt)
        print('QW.shape',QW.shape)
        alpha = torch.softmax(torch.matmul(QW, K) , dim=-1)
        print('alpha.shape',alpha.shape)
        print('restEncoding.shape',restEncoding.shape)
        print(alpha[0,0,:])

        attention = torch.matmul(alpha, restEncoding).squeeze(-2)
        attention = self.attentionMLP(attention)

        output = torch.cat((testEncoding, attention), -1)
        output = self.MLP(output)


        return output



class distilFocalLoss(loss._WeightedLoss):
    """
    modified focalloss that take teacher logits as input

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        teacherTargets: A float tensor with the same shape as inputs. Stores teacher's logits for each element in inputs
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        beta: (optional) Weighting factor in range (0,1) to balance teacher's correct but student's wrong prediction vs others
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    __constants__ = ['reduction']
    def __init__(self, alpha: float = -1, beta: float = 0.75, gamma: float = 2, reduction: str = "mean"):
        super(distilFocalLoss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def get_gamma(self,p_t):
        p_t = p_t.detach()
        gammas=p_t*0
        gammas[p_t<0.5] = 3
        gammas[p_t < 0.2] = 5
        gammas[p_t>=0.5] = self.gamma
        return gammas

    def forward(self,inputs,targets,teacherLogits):
        # print('knowledge distill')
        inputs = inputs.flatten()
        targets = targets.flatten()
        teacherLogits = teacherLogits.flatten()

        p_teacher = torch.sigmoid(teacherLogits)
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        p_t_teacher = p_teacher * targets + (1 - p_teacher) * (1 - targets)
        # gamma = self.get_gamma(p_t)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # failed = ((p_t_teacher > p_t) &(p_t<0.9)).detach()
        # if len(failed)<1000:
        #     print('# failed cell-type specific cases',torch.sum(failed),p_t[failed])
        idx = (p_t_teacher > p_t).detach()
        loss[idx] = loss[idx]*self.beta
        idx = (p_t_teacher <= p_t).detach()
        loss[idx] = loss[idx] * (1-self.beta)


        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
# class probFocalLoss(loss._WeightedLoss):
#         """
#         modified focalloss that take teacher logits as input
#
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#             alpha: (optional) Weighting factor in range (0,1) to balance
#                     positive vs negative examples or -1 for ignore. Default = 0.25
#             gamma: Exponent of the modulating factor (1 - p_t) to
#                    balance easy vs hard examples.
#             reduction: 'none' | 'mean' | 'sum'
#                      'none': No reduction will be applied to the output.
#                      'mean': The output will be averaged.
#                      'sum': The output will be summed.
#         Returns:
#             Loss tensor with the reduction option applied.
#         """
#         __constants__ = ['reduction']
#
#         def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
#             super().__init__()
#             self.alpha = alpha
#             self.gamma = gamma
#             self.reduction = reduction
#
#
#         def forward(self, p, targets):
#             p = p.flatten()
#             targets = targets.flatten()
#             ce_loss = F.binary_cross_entropy(p, targets, reduction="none")
#             p_t = p * targets + (1 - p) * (1 - targets)
#             loss = ce_loss * ((1 - p_t) ** self.gamma)
#
#             if self.alpha >= 0:
#                 alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#                 loss = alpha_t * loss
#
#             if self.reduction == "mean":
#                 loss = loss.mean()
#             elif self.reduction == "sum":
#                 loss = loss.sum()
#
#             return loss

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}