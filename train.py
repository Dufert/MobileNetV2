#
# Created by orange on 2021/3/3.
#

import os
import glog as log
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as tf

from anyDataSetClass import anyDataSetClass
from mobileNetV2Class import mobileNetV2Class


def train(train_data_loader, loss_fn, optim, data_set_number):
    data_number = 0
    true_number = 0
    train_losses = 0
    for train_data, train_label in train_data_loader:
        train_data = train_data.to(device)
        train_label = train_label.to(device)

        data_number += train_label.shape[0]
        optim.zero_grad()

        pred = model(train_data)

        loss = loss_fn(pred, train_label)
        train_losses += loss.item()

        _, pred_result = torch.max(pred, 1)
        true_number += (pred_result == train_label).sum().item()

        loss.backward()
        optim.step()

        if data_number % 4000 == 0:
            log.info("epoch : {} {}/{} train loss {} train accuracy {}".format(epoch,
                                                                               data_number,
                                                                               data_set_number,
                                                                               round(train_losses / data_number, 6),
                                                                               round(true_number / data_number, 6)))
    log.info("Epoch : {} Train Data Set Loss {} Accuracy {}".format(epoch,
                                                                    round(train_losses / data_number, 6),
                                                                    round(true_number / data_number, 6)))

    return true_number / data_number, train_losses / data_number


def validation(val_data_loader, data_set_number):
    with torch.no_grad():
        val_losses = 0
        val_true_number = 0
        for val_data, val_label in val_data_loader:
            val_data = val_data.to(device)
            val_label = val_label.to(device)

            val_pred = model(val_data)

            val_losses += loss_fn(val_pred, val_label).item()

            _, val_result = torch.max(val_pred, 1)
            val_true_number += torch.sum(val_result == val_label).item()

        val_loss_node = val_losses / data_set_number
        val_acc_node = val_true_number / data_set_number

        log.info("Epoch : {} Val Data Set Loss : {} Accuracy {}".format(epoch,
                                                                        round(val_loss_node, 6),
                                                                        round(val_acc_node, 6)))

        return val_acc_node, val_loss_node


train_tf = tf.Compose([tf.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
                       tf.RandomHorizontalFlip(0.5),
                       tf.ToTensor(),
                       tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                       ])

cv_tf = tf.Compose([tf.Resize((224, 224)),
                    tf.ToTensor(),
                    tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    ])

train_dataset = anyDataSetClass(root="/home/orange/DufertWork/Linkfile/Library/objdetectionDataSet/train", tf=train_tf)
cv_dataset = anyDataSetClass(root="/home/orange/DufertWork/Linkfile/Library/objdetectionDataSet/val", tf=cv_tf)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True)
cv_data_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=16, num_workers=1, shuffle=True)

log.info("train data set size:{}".format(train_dataset.__len__()))
log.info("cv data set size:{}".format(cv_dataset.__len__()))

class_name = list(train_dataset.class_name)
log.info("class name is {}".format(class_name))

# model init module
model_path = "mv2_pth.tar"
suffix_path = "model"

classes = len(class_name)
save_model = True
start_epoch = 0
epochs = 100
epochs_since_improvement = 0
best_loss = 1e+3
best_acc = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(suffix_path):
    os.mkdir(suffix_path)

if os.path.exists(os.path.join(suffix_path, model_path)):
    checkpoint = torch.load(os.path.join(suffix_path, model_path))
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_loss = checkpoint['best_loss']
    lr_scheduler = checkpoint['lr_scheduler']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    log.info("Loaded checkpoint from epoch: {} best loss: {} epochs_since_improvement: {}".format(start_epoch - 1,
                                                                                                  round(best_loss, 6),
                                                                                                  epochs_since_improvement))
else:
    model = mobileNetV2Class(2, pretrain=True)
    # 构建learning schedule

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)
    log.info("create model")

loss_fn = nn.CrossEntropyLoss()

ta_tl_va_vl = []

for epoch in range(start_epoch, epochs, 1):
    pass

    log.info("epochs since improvement: {}".format(epochs_since_improvement))
    if epochs_since_improvement == 10:
        log.info("early stop train")
        break

    train_acc, train_loss = train(train_data_loader, loss_fn, optimizer, train_dataset.__len__())
    val_acc, val_loss = validation(cv_data_loader, cv_dataset.__len__())
    ta_tl_va_vl.append([train_acc, train_loss, val_acc, val_loss])

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_since_improvement = 0
        is_best = True
    else:
        epochs_since_improvement += 1
        is_best = False

    if save_model:
        model_dict = {"epoch": epoch,
                      "best_loss": best_loss,
                      "model": model,
                      "optim": optimizer,
                      "lr_scheduler": lr_scheduler,
                      "epochs_since_improvement": epochs_since_improvement,
                      }
        torch.save(model_dict, os.path.join(suffix_path, model_path))
        if is_best:
            torch.save(model_dict, os.path.join(suffix_path, "Best_" + model_path))

ta_tl_va_vl = np.array(ta_tl_va_vl)

plt.title('loss and accuracy')
for i in range(ta_tl_va_vl.shape[1]):
    plt.plot(ta_tl_va_vl[:, i])
plt.xlabel('epoch')
plt.show()
