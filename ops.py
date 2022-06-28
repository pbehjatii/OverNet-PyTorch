

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        self.SE = SE(64, reduction=16)
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(64, 64*expand, 1, padding=1//2)))
        body.append(nn.ReLU(inplace=True))
        body.append(
            wn(nn.Conv2d(64*expand, int(64*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(64*linear), 64, 3, padding=3//2)))
        self.body = nn.Sequential(*body)


    def forward(self, x):

        out = self.body(x)
        out = self.SE(out)
        out = self.res_scale(out) + self.x_scale(x)
        return out



class BasicConv2d(nn.Module):
    def __init__(self, wn, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True))

        self.LR = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.LR(x)
        return x



class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, upscale, wn, group=1):
        super(UpsampleBlock, self).__init__()

        self.up = _UpsampleBlock(n_channels, upscale=upscale, wn=wn, group=group)


    def forward(self, x, upscale):
        return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, upscale, wn,  group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []

        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                modules += [wn(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)),
                            nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]

        elif upscale == 3:
            modules += [wn(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group)),
            nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        elif upscale == 5:
            modules += [wn(nn.Conv2d(n_channels, 25 * n_channels, 3, 1, 1, groups=group)),
            nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(5)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out
