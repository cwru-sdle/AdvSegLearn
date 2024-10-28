import unittest
import torch
import sys
import os
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from multiclass_dataset import Multiclass_dataset

class TestMulticlass_dataset(unittest.TestCase):
    
    def test_get_device(self):
        dataset = Multiclass_dataset()
        device = dataset.get_device()
        self.assertIn(device.type, ['cpu', 'cuda'])
        
    def test_imgs_to_channel(self):
        dataset = Multiclass_dataset(IMG_SIZE=[256, 256])
        img_paths = ['path/to/image1.png', 'path/to/image2.png']
        # Mock the imread and resize functions for testing without actual files
        def mock_imread(path, as_gray):
            return np.random.rand(100, 100)
        def mock_resize(image, output_shape):
            return np.random.rand(output_shape[0], output_shape[1])
        dataset.imread = mock_imread
        dataset.resize = mock_resize
        imgs_tensor = dataset.imgs_to_channel(img_paths)
        self.assertEqual(imgs_tensor.shape, torch.Size([2, 256, 256]))
    
    def test_validate_supervised_data(self):
        dataset = Multiclass_dataset()
        x_sup = [['img1', 'img2'], ['img3', 'img4']]
        y = [['label1', 'label2'], ['label3', 'label4']]
        try:
            dataset.validate_supervised_data(x_sup, y)
        except IndexError as e:
            self.fail(f"validate_supervised_data raised IndexError unexpectedly: {e}")
    
    def test_initialize_data(self):
        x_sup = [['img1', 'img2'], ['img3', 'img4']]
        y = [['label1', 'label2'], ['label3', 'label4']]
        x_unsup = [['img5', 'img6'], ['img7', 'img8']]
        dataset = Multiclass_dataset(x_sup=x_sup, y=y, x_unsup=x_unsup)
        self.assertEqual(len(dataset), 4)
    
    def test_len(self):
        dataset = Multiclass_dataset()
        self.assertEqual(len(dataset), 0)
        x_sup = [['img1'], ['img2']]
        y = [['label1'], ['label2']]
        dataset = Multiclass_dataset(x_sup=x_sup, y=y)
        self.assertEqual(len(dataset), 1)
        
    def test_getitem(self):
        x_sup = [['img1'], ['img2']]
        y = [['label1'], ['label2']]
        dataset = Multiclass_dataset(x_sup=x_sup, y=y)
        img, label = dataset[0]
        self.assertEqual(img.shape, torch.Size([2, 512, 512]))
        self.assertEqual(label.shape, torch.Size([2, 512, 512]))

if __name__ == '__main__':
    unittest.main()
