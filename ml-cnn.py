import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 构建多标签CNN模型
class MLCnn(nn.Module):
    '''
    label_type_size - 标签类别的数量
    '''
    def __init__(self, label_type_size):
        # 调用模型的父类构造函数
        super(MLCnn, self).__init__()

        # 创建网络层
        self.layer = nn.Sequential(
            # 创建一个全链接层，把2维数据转化为64维数据
            nn.Linear(2, 64),
            # 创建ReLU激活函数
            nn.ReLU(),
            # 创建一个全链接层，把64维数据转化为标签数量
            nn.Linear(64, label_type_size)
        )

    '''
    创建向前传播
    '''
    def forward(self, input_x):
        return self.layer(input_x)


if __name__ == '__main__':
    # 训练数据集
    train = []
    # 训练集标签
    labels = []

    # 随机创建一组多标签数据集和
    for i in range(10000):
        category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
        if category == (1, 0):
            train.append([np.random.uniform(0.1, 1), 0])
            labels.append([1, 0, 1])
        if category == (0, 1):
            train.append([0, np.random.uniform(0.1, 1)])
            labels.append([0, 1, 0])
        if category == (0, 0):
            train.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
            labels.append([0, 0, 1])
        if category == (1, 1):
            train.append([0, 0])
            labels.append([1, 0, 0])

    # 获取标签数量
    label_type_size = len(labels[0])
    # 创建分类模型
    clf = MLCnn(label_type_size)

    # 创建参数优化器
    optimizer = optim.Adam(clf.parameters())
    # 创建多标签的损失函数
    criterion = nn.MultiLabelSoftMarginLoss()

    # 设置迭代次数
    epochs = 5
    # 开始循环迭代
    for epoch in range(epochs):
        # 保存训练过程的损失函数
        losses = []
        for i, sample in enumerate(train):
            # 创建Tensor对象
            sample_x = torch.tensor(sample, dtype=torch.float, requires_grad=True).view(1, -1)
            label_y = torch.tensor(labels[i], dtype=torch.float, requires_grad=True).view(1, -1)

            # 创建分类器，并运行一次前向网络
            output = clf(sample_x)
            # 输出和标签进行对比
            loss = criterion(output, label_y)

            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 一步随机梯度下降
            optimizer.step()
            # 添加损失均值
            losses.append(loss.data.mean())

        loss_mean = np.mean(losses)
        print('[{:d}/{:d}] Loss: {:.3f}'.format((epoch + 1), epochs, loss_mean))