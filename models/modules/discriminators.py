import argparse
import functools

import numpy as np
from torch import nn
from torch.nn import functional as F

from models.networks import BaseNetwork

class FLAGS(object):
    teacher_ids = 1

class NLayerDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class MultiNLayerDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, n_share, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        self.n_share = n_share
        super(MultiNLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        block1s = []
        block2s = []
        block3s = []
        block4s = []
        block5s = []

        for _ in [0, 1]:
            block1s.append(ConvReLU(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
            block2s.append(ConvBNReLU(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer, use_bias=use_bias))
            block3s.append(ConvBNReLU(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer, use_bias=use_bias))
            block4s.append(ConvBNReLU(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, norm_layer=norm_layer, use_bias=use_bias))
            block5s.append(Conv(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw))

        self.block1s = nn.ModuleList(block1s)
        self.block2s = nn.ModuleList(block2s)
        self.block3s = nn.ModuleList(block3s)
        self.block4s = nn.ModuleList(block4s)
        self.block5s = nn.ModuleList(block5s)

    def forward(self, input):
        idx = 0 if FLAGS.teacher_ids == 1 else 1

        h = input
        h = self.block1s[-1 if self.n_share > 0 else idx](h)
        h = self.block2s[-1 if self.n_share > 1 else idx](h)
        h = self.block3s[-1 if self.n_share > 2 else idx](h)
        h = self.block4s[-1 if self.n_share > 3 else idx](h)
        output = self.block5s[-1 if self.n_share > 4 else idx](h)
        return output

class PixelDiscriminator(BaseNetwork):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class ConvBNReLU(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            norm_layer=nn.BatchNorm2d,
            use_bias=True):
        super(ConvBNReLU, self).__init__()

        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=use_bias),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.block(x)
        return x

class ConvReLU(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            use_bias=True):
        super(ConvReLU, self).__init__()

        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.block(x)
        return x

class Conv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            use_bias=True):
        super(Conv, self).__init__()

        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=use_bias),
            )

    def forward(self, x):
        x = self.block(x)
        return x