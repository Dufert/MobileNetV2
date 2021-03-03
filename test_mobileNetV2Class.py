#
# Created by orange on 2021/3/3.
#
import torch
import unittest
import torchvision.models as models
from mobileNetV2Class import mobileNetV2Class


class TestmobileNetV2Class(unittest.TestCase):
    def test_shape(self):
        model = mobileNetV2Class(num_classes=2, pretrain=True)
        std_model = models.MobileNetV2(num_classes=2)

        x = torch.randn((2, 3, 224, 224), dtype=torch.float32)

        y_std_model = std_model(x)
        y_std_features = std_model.features(x)

        y_model = model(x)
        y_features = model.features(x)

        self.assertEqual(list(y_model.shape), list(y_std_model.shape))
        self.assertEqual(list(y_features.shape), list(y_std_features.shape))

if __name__ == '__main__':
    unittest.main()
