#
# Created by orange on 2021/3/3.
#
import unittest
import torchvision.transforms as tf

from anyDataSetClass import anyDataSetClass

class TestanyDataSetClassGetItem(unittest.TestCase):
    def test_GetItem(self):
        path = "/home/orange/DufertWork/Linkfile/Library/objdetectionDataSet/val"
        tfsm = tf.Compose([tf.ToTensor()])
        dataset = anyDataSetClass(path, tfsm)

        data, label = dataset.__getitem__(250)
        self.assertEqual(label, 1)

        data, label = dataset.__getitem__(0)
        self.assertEqual(label, 0)

        self.assertGreater(dataset.__len__(), 0)

if __name__ == '__main__':
    unittest.main()