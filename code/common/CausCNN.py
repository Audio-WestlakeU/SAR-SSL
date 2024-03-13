""" 
    Function:  Commenly used causal CNNs 
    Refs:  
"""

import torch.nn as nn


class CausConv3d(nn.Module):
    """ Causal 3D Convolution for SRP-PHAT maps sequences
	"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausConv3d, self).__init__()
        self.pad = kernel_size[0] - 1
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=(self.pad, 0, 0))

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad, :, :]


class CausConv2d(nn.Module):
    """ Causal 2D Convolution for spectrograms and GCCs sequences
	"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausConv2d, self).__init__()
        self.pad = kernel_size[0] - 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad, :]


class CausConv1d(nn.Module):
    """ Causal 1D Convolution
	"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad]


class CausCnnBlock1x1(nn.Module):
    # expansion = 1
    def __init__(self, inplanes, planes, kernel=(1,1), stride=(1,1), padding=(0,0)):
        super(CausCnnBlock1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv1(x)

        return out

class CnnBlock(nn.Module):
    """ Function: Basic convolutional block
        reference: resnet, https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
    """
    # expansion = 1
    def __init__(self, inplanes, planes, kernel=(3,3), stride=(1,1), padding=(1,1), use_res=True, downsample=None):
        super(CnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_res == True:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

class CausCnnBlock(nn.Module):
    """ Function: Basic causal convolutional block
	"""
    # expansion = 1
    def __init__(self, inplanes, planes, kernel=(3,3), stride=(1,1), padding=(1,2), use_res=True, downsample=None):
        super(CausCnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # cannot use with gradients accumulation
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.pad = padding
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.pad[1] != 0:
            out = out[:, :, :, :-self.pad[1]]

        out = self.conv2(out)
        out = self.bn2(out)
        if self.pad[1] != 0:
            out = out[:, :, :, :-self.pad[1]]

        if self.use_res == True:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out