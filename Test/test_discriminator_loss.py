import unittest
import torch
from torch.testing import assert_close
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from discriminator_loss import discriminator_loss

class TestDiscriminatorLoss(unittest.TestCase):
    
    def setUp(self):
        self.loss_fn = discriminator_loss()

    def test_shape_check(self):
        # Test tensor with incorrect shape
        with self.assertRaises(IndexError):
            self.loss_fn(torch.randn(2, 3, 4, 5, 6), ground_truth=True)

    def test_ground_truth_true(self):
        # Test with ground_truth=True
        disc_output = torch.tensor([[[[0.9, 0.8], [0.7, 0.6]]]], dtype=torch.float32)
        expected_loss = -torch.sum(torch.log(disc_output + 1e-8))
        actual_loss = self.loss_fn(disc_output, ground_truth=True)
        assert_close(actual_loss, expected_loss)

    def test_ground_truth_false(self):
        # Test with ground_truth=False
        disc_output = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32)
        expected_loss = -torch.sum(torch.log(1 - disc_output + 1e-8))
        actual_loss = self.loss_fn(disc_output, ground_truth=False)
        assert_close(actual_loss, expected_loss)

    def test_small_values(self):
        # Test with small values
        disc_output = torch.full((1, 2, 2, 2), 1e-10, dtype=torch.float32)
        expected_loss = -torch.sum(torch.log(disc_output + 1e-8))
        actual_loss = self.loss_fn(disc_output, ground_truth=True)
        assert_close(actual_loss, expected_loss)

if __name__ == '__main__':
    unittest.main()