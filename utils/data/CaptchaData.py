from utils.data.IDataParse import IDataParse


# 4位验证码数据
class CaptchaData(IDataParse):
    def __init__(self, data_path=None):
        super().__init__(data_path)

    def parse_train_data(self):
        super().parse_train_data()

    def parse_validation_data(self):
        super().parse_validation_data()

    def parse_test_target(self):
        super().parse_test_target()