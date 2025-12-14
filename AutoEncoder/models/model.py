import torch.nn as nn
import numpy as np
import torchvision
import torch
from einops import rearrange

from AutoEncoder.models.merit_lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1, bias=False):
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

        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
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


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x



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


class DCG(nn.Module):
    def  __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, out_features*2)
        self.conv = DeformableConv2d(in_features,out_features)#2.7
        self.act = act_layer()

    def forward(self, z):
        B,N,C=z.shape
        H,W=int(N**0.5),int(N**0.5)
        x, v = self.fc1(z).chunk(2, dim=-1)
        x =x.permute(0, 2, 1).reshape(B, C, H, W)
        self.conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        x = self.act(x) * v
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Linear(d_model,d_model)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DCG(d_model)
        self.proj_2 = nn.Linear(d_model,d_model)

    def forward(self, x):
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, norm_layer=nn.LayerNorm):
        super().__init__()
        out_dim = in_out_chan
        x1_dim = in_out_chan
        self.input_size=input_size
        self.decoder_block=DecoderBlock(x1_dim)
        self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
        self.init_weights()

    def forward(self, x1, x2=None):
        x=x1
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x = x1 + x2
        x=self.decoder_block(x)
        out = self.layer_up(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class FinalDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, norm_layer=nn.LayerNorm):
        super().__init__()
        out_dim = in_out_chan
        x1_dim = in_out_chan
        self.decoder_block= DecoderBlock(x1_dim)
        self.last_layer = nn.Conv2d(out_dim, 1, 1)
        self.act=nn.Sigmoid()
        self.init_weights()

    def forward(self, x1, x2=None):
        b, h, w, c = x2.shape  
        x2 = x2.view(b, -1, c)
        x = x1 + x2
        x = self.decoder_block(x)
        out=self.last_layer(x.view(b,  h,  w, -1).permute(0, 3, 1, 2))
        out = self.act(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class Encoder_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.backbone = maxxvit_rmlp_small_rw_256_4out()
        # Decoder
        feat_size = 32
        in_out_chan = [32, 64, 128,256]
        self.decoder_3 = DecoderLayer((feat_size, feat_size),in_out_chan[3])
        self.decoder_2 = DecoderLayer((feat_size * 2, feat_size * 2),in_out_chan[2])
        self.decoder_1 = DecoderLayer((feat_size * 4, feat_size * 4),in_out_chan[1])
        self.decoder_0 = FinalDecoderLayer((feat_size * 8, feat_size * 8),in_out_chan[0])

    def encoder(self,x):
        return self.backbone(x)

    def decoder(self,x):
        b, c, _, _ = x[3].shape
        tmp_3 = self.decoder_3(x[3].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, x[2].permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, x[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, x[0].permute(0, 2, 3, 1))
        return tmp_0

    def forward(self, x):
        en_x=self.encoder(x)
        return self.decoder(en_x)



