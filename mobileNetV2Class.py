#
# Created by orange on 2021/3/3.
#
import torch
import torch.nn as nn
import torchvision.models as models
models.MobileNetV2

class mobileNetV2Class(nn.Module):
    # net 子结构



    def __init__(self, classes):
        super(mobileNetV2Class, self).__init__()    # _construct and training 父类属性没有使用到

        self.classes = classes



        # Params from paper
        param_list = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

    def forward(self, input):
        input = self.features(input)

        #to Linear layer shape


        output = self.Linear(input)

        return output

