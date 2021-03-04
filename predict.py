#
# Created by orange on 2021/3/3.
#

import os
import cv2
import numpy as np
import glog as log
import PIL.Image as Image

import torch
import torchvision.transforms as tf

if __name__ == "__main__":
    path = "model/Best_mv2_pth.tar"
    if not os.path.exists(path):
        log.error("checkpoint is not exist")
        exit(0)
    test_path = "/media/orange/anoDisk/workspace/Library/objdetectionDataSet/val/exist"
    if not os.path.isdir(test_path):
        log.info("test path no dir")
        exit(0)
    img_list = os.listdir(test_path)
    if len(img_list) == 0:
        log.error("test set is empty")
        exit(0)

    classes = ["exist", "noexist"]
    test_tf = tf.Compose([tf.Resize((224, 224)),
                          tf.ToTensor(),
                          tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                          ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.eval()
    model.to(device)

    cv2.namedWindow("predict", cv2.WINDOW_NORMAL)
    for img_name in img_list:
        img = Image.open(os.path.join(test_path, img_name))
        img = img.convert("RGB")

        show_img = np.uint8(img)

        img = test_tf(img)
        img.to(device)

        img = img.unsqueeze(0)  # 将输入改成1*shape

        y_pred = model(img)
        _, result = torch.max(y_pred, 1)  # torch max 第二项 0为每列的最大值, 1为每行的最大值, 返回第一个为最大值,第二个为最大值的索引

        log.info(classes[result.item()])  # result

        if result.item() == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(show_img, classes[result.item()], (20, 50), 1, 1, color, 1)
        cv2.imshow("predict", show_img)
        k = cv2.waitKey(20)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
