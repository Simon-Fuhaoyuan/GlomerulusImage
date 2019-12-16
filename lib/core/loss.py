# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=True)

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss
