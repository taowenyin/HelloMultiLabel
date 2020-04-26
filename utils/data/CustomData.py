from utils.data.IDataParse import IDataParse

import numpy as np


# 自定义数据
class CustomData(IDataParse):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    '''
    size: 生成数据的大小
    '''
    def __generate_data(self, size):
        # 生成的数据集
        data_values = []
        # 生成的标签集
        data_labels = []

        for i in range(size):
            # 生成数据的类型
            category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
            if category == (1, 0):
                data_values.append([np.random.uniform(0.1, 1), 0])
                data_labels.append([1, 0, 1])
            if category == (0, 1):
                data_values.append([0, np.random.uniform(0.1, 1)])
                data_labels.append([0, 1, 0])
            if category == (0, 0):
                data_values.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
                data_labels.append([0, 0, 1])
            if category == (1, 1):
                data_values.append([0, 0])
                data_labels.append([1, 0, 0])

        return data_labels, data_values

    def parse_train_data(self):
        return self.__generate_data(10000)

    def parse_validation_data(self):
        return self.__generate_data(5000)

    def parse_test_target(self):
        return self.__generate_data(5000)