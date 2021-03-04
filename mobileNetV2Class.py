#
# Created by orange on 2021/3/3.
#

import torchvision
import torch.nn as nn


# net 子结构
class ConvBnReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, strides, kernel_size, groups):  # 重载
        padding = kernel_size // 2  # 向下取整
        super(ConvBnReLU, self).__init__(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=padding, kernel_size=kernel_size, stride=strides,
                      groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU6(inplace=True)
        )


# net 子结构 瓶颈层
class bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, strides, expand_ratio):
        super(bottleneck, self).__init__()
        assert strides in [1, 2]  # 避免错误的stride

        hidden_dims = int(in_ch * expand_ratio)  # expansion layer output
        self.addFlag = in_ch == out_ch and strides == 1

        layer = []
        # 如果扩展等于1则不需要加expansion layer
        if expand_ratio != 1:
            layer.append(ConvBnReLU(in_ch=in_ch, out_ch=hidden_dims, strides=1, kernel_size=1, groups=1))
        # dw,深度可分离卷积
        layer.append(
            ConvBnReLU(in_ch=hidden_dims, out_ch=hidden_dims, strides=strides, kernel_size=3, groups=hidden_dims))
        # pw,projection layers-linear,由高维到低维度,避免ReLU导致非线性特征信息损失
        layer.append(nn.Conv2d(in_channels=hidden_dims, out_channels=out_ch, kernel_size=1, stride=1, bias=False))
        layer.append(nn.BatchNorm2d(num_features=out_ch))

        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        if self.addFlag:
            return x + self.conv(x)
        return self.conv(x)


class mobileNetV2Class(nn.Module):

    def __init__(self, num_classes, param_list=None, pretrain=False):
        super(mobileNetV2Class, self).__init__()  # _construct and training 父类属性没有使用到

        self.classes = num_classes
        self.input_channels = 32
        self.output_channels = 1280

        # Params from paper

        if param_list is None:
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

        feature = []
        feature.append(ConvBnReLU(in_ch=3, out_ch=self.input_channels, strides=2, kernel_size=3, groups=1))

        for (t, c, n, s) in param_list:
            for i in range(n):
                s = s if i == 0 else 1
                feature.append(bottleneck(in_ch=self.input_channels, out_ch=c, strides=s, expand_ratio=t))
                self.input_channels = c

        feature.append(
            ConvBnReLU(in_ch=self.input_channels, out_ch=self.output_channels, strides=1, kernel_size=1, groups=1))
        # feature.append(nn.AvgPool2d(kernel_size=7))# 使用mean代替,方便单元测试时计算其shape
        self.features = nn.Sequential(*feature)

        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=self.output_channels, out_features=self.classes)
        )

        # weight initialization
        if pretrain is False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            self._load_pretrain_params()
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def _load_pretrain_params(self):
        state_dict = self.state_dict()

        mv2dict = torchvision.models.mobilenet_v2(pretrained=True).state_dict()
        params = list(mv2dict.keys())
        params = params[:-2]  # 由于classes不一样故,最后的classifier不可以用preTrain 值,如实在是想用可以进行采样
        # print(list(state_dict.keys()))
        # print(params)

        for i, params_name in enumerate(params):
            state_dict[params_name] = mv2dict[params[i]]

        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # shape [n 1280 7 7] to [n 1280]
        y = self.classifier(x)
        return y


# if __name__ == "__main__":
#     model = mobileNetV2Class(2, pretrain=True)
