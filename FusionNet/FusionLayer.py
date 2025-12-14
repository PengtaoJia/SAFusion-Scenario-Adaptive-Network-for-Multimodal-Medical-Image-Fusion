import torch
import torch.nn as nn
from FusionNet.MMoE import MMoE
import torch.nn.functional as F
import numpy as np
from FusionNet.SGFM import MultiTaskGeneralFusion
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last == False:
            out = F.relu(out, inplace=True)
        return out


class FusionBlock(nn.Module):

    def __init__(self, channel=1024, r=16, task_num=3):
        super().__init__()
        self.channel = channel
        self.fc = nn.Linear(channel * 2, channel, bias=False)
        self.norm = nn.LayerNorm(channel)
        self.act = nn.ReLU()
        self.fusion1 = MultiTaskGeneralFusion(channel, r)
        self.fusion2 = MMoE(channel, channel, 4, channel // 2, noisy_gating=True, k=2, task_num=task_num)

        self.init_scale_shift()

    def init_scale_shift(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, t, task_index):
        B, C, H, W = x.shape
        N = H * W
        # SGFM
        z = self.fusion1(x, t)

        # SSFM
        x = to_3d(x)
        t = to_3d(t)
        y = torch.cat([x, t], dim=-1)
        y = self.norm(self.act(self.fc(y)))
        y, aux_loss = self.fusion2(y.reshape(B * N, C), task_index)
        y = to_4d(y.reshape(B, N, C), H, W) + z

        return y, aux_loss


class Fusion_network(nn.Module):
    def __init__(self, nC=[256,256,256,256,256]):
        super(Fusion_network, self).__init__()
        self.nC=nC
        task_num=3

        self.fusion_block1 = FusionBlock(nC[0], r=256, task_num=task_num)
        self.fusion_block2 = FusionBlock(nC[1], r=128, task_num=task_num)
        self.fusion_block3 = FusionBlock(nC[2], r=64, task_num=task_num)
        self.fusion_block4 = FusionBlock(nC[3], r=32, task_num=task_num)


    def forward(self, x1, x2,type=0):
        f1_0, a = self.fusion_block1(x1[0], x2[0], type)
        f2_0, b = self.fusion_block2(x1[1], x2[1], type)
        f3_0, c = self.fusion_block3(x1[2], x2[2], type)
        f4_0, d = self.fusion_block4(x1[3], x2[3], type)
        return [f1_0, f2_0, f3_0, f4_0],a+b+c+d


class CNNBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(CNNBlock, self).__init__()
        dim = in_channels
        self.output = nn.Sequential(
        nn.Conv2d(int(dim) * 2, int(dim), kernel_size=3,
                  stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(int(dim), dim, kernel_size=3,
                  stride=1, padding=1, ), )
    def forward(self, x, y):
        return self.output(torch.cat([x, y], dim=1))



