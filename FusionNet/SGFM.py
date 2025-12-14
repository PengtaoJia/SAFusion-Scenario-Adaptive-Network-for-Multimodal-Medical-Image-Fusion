import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

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

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class DCG(nn.Module):
    def  __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features*2)
        self.conv = DeformableConv2d(in_features,out_features)#2.7
        self.act = act_layer()

    def forward(self, z):
        z=z.reshape(z.shape[0],z.shape[1],-1).permute(0,2,1)
        B,N,C=z.shape
        H,W=int(N**0.5),int(N**0.5)
        x, v = self.fc1(z).chunk(2, dim=-1)

        x =x.permute(0, 2, 1).reshape(B, C, H, W)
        self.conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        x = self.act(x) * v

        return x.permute(0,2,1).reshape(B,C,H,W)



class MultiTaskGeneralFusion(nn.Module):
    def __init__(self, channels,r=0):
        super(MultiTaskGeneralFusion, self).__init__()
        self.conv_fusion = ConvLayer(2 * channels, channels, 1, 1)
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.conv2 = ConvLayer(channels, channels, 3, 1)

        block = []
        block += [ConvLayer(2 * channels, channels, 1, 1),
                  ConvLayer(channels, channels, 3, 1),
                  ConvLayer(channels, channels, 3, 1)
                  ]
        self.bottelblock = nn.Sequential(*block)
        self.dcg = DCG(channels)

    def forward(self, x1, x2):

        f_cat = torch.cat([x1, x2], 1)
        f_init = self.conv_fusion(f_cat)
        f_init = self.dcg(f_init)
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)

        out = torch.cat([out1, out2], 1)
        out = self.bottelblock(out)
        out = f_init + out

        return out


