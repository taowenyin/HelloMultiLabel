import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


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
    train_values = []
    # 测试数据集
    test_values = []
    # 训练集标签
    train_labels = []
    # 测试集标签
    test_labels = []
    # 保存损失平均值
    losses_mean = []
    # 每批数据的大小
    batch_size = 1000

    # 随机创建一组多标签训练数据集
    for i in range(10000):
        category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
        if category == (1, 0):
            train_values.append([np.random.uniform(0.1, 1), 0])
            train_labels.append([1, 0, 1])
        if category == (0, 1):
            train_values.append([0, np.random.uniform(0.1, 1)])
            train_labels.append([0, 1, 0])
        if category == (0, 0):
            train_values.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
            train_labels.append([0, 0, 1])
        if category == (1, 1):
            train_values.append([0, 0])
            train_labels.append([1, 0, 0])
    # 创建训练数据集
    train_dataset = TensorDataset(
        torch.tensor(train_values, dtype=torch.float, requires_grad=True),
        torch.tensor(train_labels, dtype=torch.float, requires_grad=True))

    # 随机创建一组多标签测试数据集
    for i in range(10000):
        category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
        if category == (1, 0):
            test_values.append([np.random.uniform(0.1, 1), 0])
            test_labels.append([1, 0, 1])
        if category == (0, 1):
            test_values.append([0, np.random.uniform(0.1, 1)])
            test_labels.append([0, 1, 0])
        if category == (0, 0):
            test_values.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
            test_labels.append([0, 0, 1])
        if category == (1, 1):
            test_values.append([0, 0])
            test_labels.append([1, 0, 0])
    # 创建训练数据集
    test_dataset = TensorDataset(
        torch.tensor(test_values, dtype=torch.float, requires_grad=True),
        torch.tensor(test_labels, dtype=torch.float, requires_grad=True))

    # 创建训练数据加载器，并且设置每批数据的大小，以及每次读取数据时随机打乱数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 把数据集分为验证即和测试集
    indices = range(len(test_dataset))
    indices_val = indices[:5000]
    indices_test = indices[5000:]
    # 构造数据采样器
    sampler_val = SubsetRandomSampler(indices_val)
    sampler_test = SubsetRandomSampler(indices_test)
    # 创建验证集加载器
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_val)
    # 测试集加载器
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_test)

    # 获取标签数量
    label_type_size = len(train_labels[0])
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
        losses_mean.append(loss_mean)
        print('[{:d}/{:d}] Loss: {:.3f}'.format((epoch + 1), epochs, loss_mean))

    # 绘制图像
    plt.plot(range(len(losses_mean)), losses_mean)
    plt.ylabel('Loss')
    plt.show()