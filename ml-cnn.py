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
    batch_size = 100

    # 训练时的损失函数
    record_train_loss = []
    # 验证时的损失值
    record_val_loss = []
    # 测试时的损失值
    record_test_loss = []

    # 预测结果
    predict = []

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
    epochs = 50
    # 开始循环迭代
    for epoch in range(epochs):
        # 训练时的损失值
        train_loss = []
        # 验证时的损失值
        val_loss = []
        # 通过enumerate获取每批次索引进行训练
        for batch_index, (data, target) in enumerate(train_loader):
            # 标记模型在训练集上训练
            clf.train()
            # 使用模型运行一次前向网络
            output = clf(data)
            # 输出和标签进行对比
            loss = criterion(output, target)

            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 一步随机梯度下降
            optimizer.step()

            # 保存训练的损失值
            train_loss.append(loss.data)

            # 每训练10批就进行一次验证
            if batch_index % 10 == 0:
                # 标记模型在验证集上
                clf.eval()
                val_loss_arr = []
                for (val_data, val_target) in validation_loader:
                    output = clf(val_data)
                    # 输出和标签进行对比
                    one_loss = criterion(output, val_target)
                    # 保存验证集误差
                    val_loss_arr.append(one_loss.data)

                # 保存验证集的损失平均值
                val_loss_sub_mean = np.mean(val_loss_arr)
                val_loss.append(val_loss_sub_mean)

        # 获取每轮迭代的平均损失
        train_loss_mean = np.mean(train_loss)
        val_loss_mean = np.mean(val_loss)
        record_train_loss.append(train_loss_mean)
        record_val_loss.append(val_loss_mean)

        print('[{:d}/{:d}] Train Loss: {:.3f} Validation Loss: {:.3f}'.format(
                    (epoch + 1), epochs, train_loss_mean, val_loss_mean))

    # 在测试集测试
    clf.eval()
    for (test_data, test_target) in test_loader:
        predict = clf(test_data)
        # 输出和标签进行对比
        test_loss = criterion(predict, test_target)
        record_test_loss.append(test_loss.data)
    print('Test Mean Loss: {:.3f}'.format(np.mean(record_test_loss)))

    # 把大于0的设为1
    predict[predict > 0] = 1
    predict[predict < 0] = 0
    predict = predict.detach().numpy().astype(np.int)

    # 绘制图像
    plt.plot(range(len(record_train_loss)), record_train_loss, label='Train Loss')
    plt.plot(range(len(record_val_loss)), record_val_loss, label='Validation Loss')
    # 在(30, 0.45)位置绘制文字
    plt.text(30, 0.45, 'Test Mean Loss: {:.3f}'.format(np.mean(record_test_loss)),
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k',lw=1 ,alpha=0.5))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('MultiLabel CNN')
    plt.show()