import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from utils.data.CustomData import CustomData
from utils.data.SceneData import SceneData


# 构建多标签CNN模型
class MLCnn(nn.Module):
    '''
    input_shape - 输入数据的形状
    label_shape - 标签类别的形状
    drop_out - 丢弃神经元的比例
    '''
    def __init__(self, input_shape, label_shape, drop_out):
        # 调用模型的父类构造函数
        super(MLCnn, self).__init__()

        # 创建网络层
        self.layer = nn.Sequential(
            # 创建一个全链接层，把2维数据转化为64维数据
            nn.Linear(input_shape, input_shape ** 2),
            # 创建ReLU激活函数
            nn.ReLU(),
            # 创建一个全链接层，把64维数据转化为标签数量
            nn.Linear(input_shape ** 2, label_shape),
            # 创建Dropout层
            nn.Dropout(p=drop_out)
        )

    '''
    创建向前传播
    '''
    def forward(self, input_x):
        return self.layer(input_x)


if __name__ == '__main__':
    # 计算开始时间
    star = time.time()

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

    # 创建自定义数据
    # dataset = CustomData()
    dataset = SceneData('dataset/scene', type=1)
    train_labels, train_data = dataset.parse_train_data()
    test_labels, test_data = dataset.parse_test_target()
    validation_labels, validation_data = dataset.parse_validation_data()

    # 创建训练数据集
    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float, requires_grad=True),
        torch.tensor(train_labels, dtype=torch.float, requires_grad=True))
    # 创建训练数据集
    test_dataset = TensorDataset(
        torch.tensor(test_data, dtype=torch.float, requires_grad=True),
        torch.tensor(test_labels, dtype=torch.float, requires_grad=True))
    # 创建验证数据集
    validation_dataset = TensorDataset(
        torch.tensor(validation_data, dtype=torch.float, requires_grad=True),
        torch.tensor(validation_labels, dtype=torch.float, requires_grad=True))

    # 创建训练数据加载器，并且设置每批数据的大小，以及每次读取数据时随机打乱数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 创建验证集加载器
    validation_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # 测试集加载器
    test_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    # 获取标签数量
    label_shape = len(train_labels[0])
    input_shape = len(train_data[0])
    # 丢弃神经元的比例
    drop_out = 0.5
    # 创建分类模型
    clf = MLCnn(input_shape, label_shape, drop_out)

    # 创建参数优化器
    optimizer = optim.Adam(clf.parameters())
    # 创建多标签的损失函数
    criterion = nn.MultiLabelSoftMarginLoss()

    # 设置迭代次数
    epochs = 100
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

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))

    # 绘制图像
    plt.plot(range(len(record_train_loss)), record_train_loss, label='Train Loss')
    plt.plot(range(len(record_val_loss)), record_val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('MultiLabel CNN Drop Out: {:.1f} Test Mean Loss: {:.3f}'.format(drop_out, np.mean(record_test_loss)))
    plt.show()