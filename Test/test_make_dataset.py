import unittest
import torch
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from multiclass_dataset import Multiclass_dataset  # Assuming Multiclass_dataset is implemented in multiclass_dataset.py
from make_dataset import dataset_setup

class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        # Setting up mock data or any necessary initialization
        self.supervised_raw = torch.randn(100, 3, 32, 32)  # Example raw data tensor (100 samples, 3 channels, 32x32)
        self.supervised_masks = torch.randint(0, 3, (100, 1, 32, 32))  # Example masks tensor (100 samples, single channel, 32x32)

    def test_dataset_setup(self):
        # Test the make_dataset function
        train_ds = dataset_setup(self.supervised_raw, self.supervised_masks, 0.1, 0.85,0.05)
        self.assertIsInstance(train_ds, Multiclass_dataset)
        # Add more specific assertions as needed based on Multiclass_dataset behavior

if __name__ == '__main__':
    unittest.main()