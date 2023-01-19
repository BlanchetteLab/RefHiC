import torch
from torch import nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn.modules import loss
from einops.layers.torch import Rearrange

class RefHiCSRNet(nn.Module):
    def __init__(self, num_feat=24, w=200):
        super(RefHiCSRNet, self).__init__()
        self.num_feat = num_feat
        self.w = w
        self.kvdim = (self.w // 2 // 2 // 2) ** 2

        self.conv01 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, 9, padding='same'),
            # nn.Dropout(0.2),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
        )
        self.maxpool11 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv12 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
        )

        self.maxpool12 = torch.nn.MaxPool2d(kernel_size=2)
        self.NNkv = nn.Sequential(
            nn.BatchNorm2d(self.num_feat),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, 1, 1),
            nn.Flatten(1)
        )

        self.convlast = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, 1, 9, padding='same'),
        )


        self.conv11to21 = nn.Conv2d(self.num_feat * 2, self.num_feat, 1)
        self.conv12to22 = nn.Conv2d(self.num_feat * 2, self.num_feat, 1)

        self.conv22 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
        )

        self.conv21 = nn.Sequential(
            nn.Conv2d(self.num_feat * 2, self.num_feat * 2, 3, padding='same'),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(self.num_feat * 2, self.num_feat, 3, padding='same'),
        )
        self.up22to21 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.attentionCNN11 = nn.Sequential(
            nn.LayerNorm((self.num_feat, self.w, self.w)),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
        )
        self.attentionCNN12 = nn.Sequential(
            nn.LayerNorm((self.num_feat, self.w // 2, self.w // 2)),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.num_feat, self.num_feat, 3, padding='same'),
        )


    def encoding(self, x, returnall=False):
        assert x.shape[-1] == self.w
        # if self.training:
        #     B,C,_,_ = x.shape
        #     x = rearrange(x, 'b c w h -> (b c) w h')
        #     x = self.hicdropout(x)
        #     x = rearrange(x, '(b c) w h -> b c w h',b=B)

        feat01 = torch.relu(self.conv01(x))
        feat11 = torch.relu(self.conv11(feat01))
        feat12 = torch.relu(self.conv12(self.maxpool11(feat11)))
        featkv = torch.relu(self.NNkv(self.maxpool12(feat12)))
        if returnall:
            return feat01, feat11, feat12, featkv
        return featkv

    def pretrain(self, x):
        featkv = self.encoding(x)
        return featkv

    def sim(self, x, y):
        sim = torch.matmul(x, y)
        return sim

    # x [b,n,w,h]: n = study sample + #reference samples
    def forward(self, x):
        # print('x.shape',x.shape)
        assert x.shape[-1] == self.w
        x = rearrange(x, 'b s c w h -> b (s c) w h') # combine sample and channel dim. #channel = 1 , #sample = 1+#reference samples
        batchSize = x.shape[0]
        x = rearrange(x, 'b s w h -> (b s) w h')[:, None, ...]
        feat01, feat11, feat12, featkv = self.encoding(x, True)

        # print(feat01.shape,feat11.shape,feat12.shape,featkv.shape, '-->')
        feat01 = rearrange(feat01, '(b s) c w h -> b s c w h', b=batchSize)
        feat11 = rearrange(feat11, '(b s) c w h -> b s c w h', b=batchSize)
        feat12 = rearrange(feat12, '(b s) c w h -> b s c w h', b=batchSize)
        featkv = rearrange(featkv, '(b s) d -> b s d', b=batchSize)
        Q = featkv[:, [0], ...]
        K = rearrange(featkv[:, 1:, ...], 'a b c -> a c b')
        V11 = rearrange(feat11[:, 1:, ...], 'b s c w h -> b s (c w h)')
        V12 = rearrange(feat12[:, 1:, ...], 'b s c w h -> b s (c w h)')

        sim = torch.matmul(Q, K)

        alpha = torch.softmax(sim / (self.kvdim ** 0.5), dim=-1)
        # print(alpha[0, ...],sim)

        attention11 = rearrange(torch.matmul(alpha, V11).squeeze(-2), 'b (c w h)-> b c w h', c=self.num_feat, w=self.w)
        attention12 = rearrange(torch.matmul(alpha, V12).squeeze(-2), 'b (c w h)-> b c w h', c=self.num_feat,
                                w=self.w // 2)

        attention11 = attention11 + torch.relu(self.attentionCNN11(attention11))
        attention12 = attention12 + torch.relu(self.attentionCNN12(attention12))

        feat11to21 = torch.relu(self.conv11to21(torch.cat((feat11[:, 0, ...], attention11), -3)))
        feat12to22 = torch.relu(self.conv12to22(torch.cat((feat12[:, 0, ...], attention12), -3)))
        feat22 = torch.relu(self.conv22(feat12to22))
        feat22to21 = self.up22to21(feat22)
        feat21 = torch.relu(self.conv21(torch.cat((feat11to21, feat22to21), -3)))
        output = torch.relu(self.convlast(feat21 + feat01[:, 0, ...]))
        return output


