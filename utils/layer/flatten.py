import torch.nn as nn


# 压平向量变为一维数据
class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, input_x):
        return input_x.view(input_x.size(0), -1)