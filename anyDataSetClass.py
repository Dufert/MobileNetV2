#
# Created by orange on 2021/3/3.
#

import os
import PIL.Image as Image
from torch.utils.data import Dataset

# data set like this:
# ├── train
#     ├── exist
#     └── noexist
# └── val
#     ├── exist
#     └── noexist


class anyDataSetClass(Dataset):
    def __init__(self, root, tf):
        self.tf = tf
        # 按照文件夹名对应class name and label
        class_name_label = {name: label for (label, name) in enumerate(os.listdir(root))}

        # 提取文件的abspath label并加到list
        self.img_path_label_list = []
        for class_name in class_name_label.keys():
            subroot = os.path.join(root, class_name)
            img_name_list = os.listdir(subroot)
            for img_name in img_name_list:
                self.img_path_label_list.append([os.path.join(subroot, img_name), class_name_label[class_name]])
        
    def __getitem__(self, item):
        # open img, 进行tf 输出data and label
        img = Image.open(self.img_path_label_list[item][0])
        img = img.convert('RGB')
        data = self.tf(img)
        label = self.img_path_label_list[item][1]

        return data, label

    def __len__(self):
        # return 数据集大小
        return len(self.img_path_label_list)
