from utils.data.IDataParse import IDataParse

from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np


'''
type：数据类型，0表示所有，1表示均值，2表示方差
data_path：数据路径
'''
class SceneData(IDataParse):
    def __init__(self, data_path=None, type=0):
        super().__init__(data_path)
        # 创建多标签二值化对象
        self.__multiLabelBinarizer = MultiLabelBinarizer()
        # 获取数据类型
        self.__type = type

        # 获取训练数据集
        df_train = pd.read_csv(data_path + '/scene_train', delim_whitespace=True, header=None)
        # 获取训练数据集
        df_test = pd.read_csv(data_path + '/scene_test', delim_whitespace=True, header=None)

        # 获取训练集的标签
        train_target = df_train.iloc[:, 0].values
        train_data = df_train.iloc[:, 1:].values
        # 获取测试集的标签
        test_target = df_test.iloc[:, 0].values
        test_data = df_test.iloc[:, 1:].values

        # 对数据进行预处理
        train_target = self.__preprocessing_target(train_target)
        test_target = self.__preprocessing_target(test_target)
        # 把标签进行多标签二值化
        self.__train_labels = self.__multiLabelBinarizer.fit_transform(train_target)
        # 把标签进行多标签二值化
        self.__test_labels = self.__multiLabelBinarizer.transform(test_target)

        # 对数据进行预处理
        train_data = self.__preprocessing_data(train_data)
        test_data = self.__preprocessing_data(test_data)

        # 根据类型检索数据
        if self.__type == 1:
            self.__train_data = train_data[:, 0:train_data.shape[1]:2]
            self.__test_data = test_data[:, 0:test_data.shape[1]:2]
        elif self.__type == 2:
            self.__train_data = train_data[:, 1:train_data.shape[1]:2]
            self.__test_data = test_data[:, 1:test_data.shape[1]:2]
        else:
            self.__train_data = train_data
            self.__test_data = test_data

    def parse_train_data(self):
        return self.__train_labels, self.__train_data

    def parse_validation_data(self):
        return self.__test_labels[:len(self.__test_labels) // 2], self.__test_data[:len(self.__test_data) // 2]

    def parse_test_target(self):
        return self.__test_labels[len(self.__test_labels) // 2:], self.__test_data[len(self.__test_data) // 2:]

    # 标签预处理，结果为N维的多数组
    def __preprocessing_target(self, target):
        # 标签集预处理
        train_target_arr = []
        for i in range(len(target)):
            item = target[i]
            if len(item) > 1:
                arr = np.array(item.split(',')).astype(np.int)
            else:
                arr = np.array([int(item)])

            # 标签进行排序
            arr = np.array(sorted(arr))
            train_target_arr.append(arr)
        # 构建标签集
        return np.array(train_target_arr)

    # 数据预处理，结果为N维的多数组
    def __preprocessing_data(self, data):
        # 保存最终的处理数据
        train_data_arr = []
        for i in range(len(data)):
            # 获取每个数据进行处理
            item = data[i]
            item_arr = []
            for j in range(len(item)):
                sub_item = np.array(item[j].split(':')).astype(np.float)
                item_arr.append(sub_item[1])

            train_data_arr.append(np.array(item_arr))
        # 构建标签集
        return np.array(train_data_arr)