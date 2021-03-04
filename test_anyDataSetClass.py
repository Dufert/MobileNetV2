#
# Created by orange on 2021/3/3.
#
import unittest
import torchvision.transforms as tf

from anyDataSetClass import anyDataSetClass


class TestanyDataSetClassGetItem(unittest.TestCase):
    def test_GetItem(self):
        path = "/home/orange/DufertWork/Linkfile/Library/objdetectionDataSet/val"
        tfsm = tf.Compose([tf.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
                           tf.RandomHorizontalFlip(0.5),
                           tf.ToTensor(),
                           tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                           ])
        dataset = anyDataSetClass(path, tfsm)

        data, label = dataset.__getitem__(250)
        self.assertEqual(label, 1)
        self.assertEqual(list(data.shape), [3, 224, 224])
        self.assertGreater(dataset.__len__(), 0)

        data, label = dataset.__getitem__(0)
        self.assertEqual(label, 0)


if __name__ == '__main__':
    unittest.main()
