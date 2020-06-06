from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from collections import OrderedDict

from torch.nn.modules.module import T_co

from utils.layer.flatten import Flatten

# 创建VGG模型
class MLVgg(nn.Module):
    def __init__(self):
        super(MLVgg, self).__init__()

        self.layer = nn.Sequential(OrderedDict([
            # 输入3个通道，输出64个通道，巻积大小为3
            ('conv1-1', nn.Conv2d(3, 64, 3)),
            ('act1-1', nn.ReLU()),
            ('conv1-2', nn.Conv2d(64, 64, 3, padding=(1, 1))),
            ('act1-2', nn.ReLU()),
            ('maxpool-1', nn.MaxPool2d((2, 2), padding=(1, 1))),

            ('conv2-1', nn.Conv2d(64, 128, 3)),
            ('act2-1', nn.ReLU()),
            ('conv2-2', nn.Conv2d(128, 128, 3, padding=(1, 1))),
            ('act2-2', nn.ReLU()),
            ('maxpool-2', nn.MaxPool2d((2, 2), padding=(1, 1))),

            ('conv3-1', nn.Conv2d(128, 256, 3)),
            ('act3-1', nn.ReLU()),
            ('conv3-2', nn.Conv2d(256, 256, 3, padding=(1, 1))),
            ('act3-2', nn.ReLU()),
            ('conv3-3', nn.Conv2d(256, 256, 3, padding=(1, 1))),
            ('act3-3', nn.ReLU()),
            ('maxpool-3', nn.MaxPool2d((2, 2), padding=(1, 1))),

            ('conv4-1', nn.Conv2d(256, 512, 3)),
            ('act4-1', nn.ReLU()),
            ('conv4-2', nn.Conv2d(512, 512, 3, padding=(1, 1))),
            ('act4-2', nn.ReLU()),
            ('conv4-3', nn.Conv2d(512, 512, 3, padding=(1, 1))),
            ('act4-3', nn.ReLU()),
            ('maxpool-4', nn.MaxPool2d((2, 2), padding=(1, 1))),

            ('conv5-1', nn.Conv2d(512, 512, 3)),
            ('act5-1', nn.ReLU()),
            ('conv5-2', nn.Conv2d(512, 512, 3, padding=(1, 1))),
            ('act5-2', nn.ReLU()),
            ('conv5-3', nn.Conv2d(512, 512, 3, padding=(1, 1))),
            ('act5-3', nn.ReLU()),
            ('maxpool-5', nn.MaxPool2d((2, 2), padding=(1, 1))),

            ('flatten', Flatten()),
            ('fc1', nn.Linear(512 * 7 * 7, 4096)),
            ('act6-1', nn.ReLU()),
            ('fc2', nn.Linear(4096, 4096)),
            ('act6-2', nn.ReLU()),
            ('fc3', nn.Linear(4096, 1000)),
            ('act6-3', nn.LogSoftmax(1)),
        ]))

    def forward(self, input_x) -> T_co:
        return self.layer(input_x)

