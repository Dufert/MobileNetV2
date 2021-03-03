#
# Created by orange on 2021/3/3.
#

import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader

import torchvision.transforms as tf

from anyDataSetClass import anyDataSetClass
from mobileNetV2Class import mobileNetV2Class

train_tf = tf.Compose([tf.Resize(224, 224),
                       tf.RandomCrop(0.9),
                       tf.RandomHorizontalFlip(0.5),
                       tf.ToTensor(),
                       tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                       ])

cv_tf = tf.Compose([tf.Resize(224, 224),
                    tf.ToTensor(),
                    tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    ])

train_dataset = anyDataSetClass(root="", tf=train_tf)