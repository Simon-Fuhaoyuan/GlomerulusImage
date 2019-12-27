from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import os
import logging

logger = logging.getLogger(__name__)

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
            is_activation=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm and is_activation:
            self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))
        elif not is_batchnorm and is_activation:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
        elif is_batchnorm and not is_activation:
            self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class pointnet(nn.Module):
    """
    Without Transform Net
    """

    def __init__(self, n_class=7, img_size=(299, 299), in_channels=3):
        super(pointnet, self).__init__()

        self.img_size = img_size
        self.conv_batchnorm_relu_1 = conv2DBatchNormRelu(3, 64, k_size=1, stride=1, padding=0)
        self.conv_batchnorm_relu_2 = conv2DBatchNormRelu(64, 64, k_size=1, stride=1, padding=0)
        self.conv_batchnorm_relu_3 = conv2DBatchNormRelu(64, 64, k_size=1, stride=1, padding=0)
        self.conv_batchnorm_relu_4 = conv2DBatchNormRelu(64, 128, k_size=1, stride=1, padding=0)
        self.conv_batchnorm_relu_5 = conv2DBatchNormRelu(128, 1024, k_size=1, stride=1, padding=0)
        
        self.global_maxpool = nn.MaxPool2d(img_size, padding=0)

        self.fc_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.fc_3 = nn.Linear(256, n_class)

    def forward(self, x):
        # img_width, img_height = x.size()[2], x.size()[3]
        # x = TNET_3(x)
        out_1 = self.conv_batchnorm_relu_1(x)
        out_2 = self.conv_batchnorm_relu_2(out_1)
        out_3 = self.conv_batchnorm_relu_3(out_2)
        # out_3 = TNET_128(out_3)
        out_4 = self.conv_batchnorm_relu_4(out_3)
        out_5 = self.conv_batchnorm_relu_5(out_4)

        out_max = self.global_maxpool(out_5)
        out_max_squeezed = out_max.squeeze()

        net = self.fc_1(out_max_squeezed)
        net = self.fc_2(net)
        net = self.fc_3(net)

        return net, out_5


def get_pose_net(cfg, is_train, **kwargs):
    model = pointnet(n_class=7)

    return model
