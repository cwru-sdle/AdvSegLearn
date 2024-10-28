import unittest
import torch
import sys
import os



# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from mixed_activation import check_mutually_exclusive, slice_and_softmax, apply_standard_activation, mixed_activation

class TestMixedActivation(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 6, 4, 4)
        self.mutually_exclusive_loss = torch.nn.Softmax(dim=0)
        self.standard_loss = torch.tanh
        self.mutually_exclusive = [{0, 1}, {2, 3}]

    def test_check_mutually_exclusive_valid(self):
        try:
            check_mutually_exclusive(self.mutually_exclusive, self.input_tensor.shape)
        except ValueError:
            self.fail("check_mutually_exclusive raised ValueError unexpectedly!")
    
    def test_check_mutually_exclusive_intersection_error(self):
        with self.assertRaises(ValueError):
            check_mutually_exclusive([{0, 1}, {1, 2}], self.input_tensor.shape)
    
    def test_check_mutually_exclusive_index_error(self):
        with self.assertRaises(ValueError):
            check_mutually_exclusive([{0, 1}, {6, 7}], self.input_tensor.shape)
    
    def test_slice_and_softmax(self):
        sets = {0, 1}
        temp = []
        tracker = [False] * self.input_tensor.shape[-3]
        temp = slice_and_softmax(self.input_tensor, sets, self.mutually_exclusive_loss, temp, 0, tracker)
        self.assertEqual(temp[0].shape[-3], len(sets))
        for batch in range(self.input_tensor.shape[-4]):
            for y_cord in range(self.input_tensor.shape[-2]):
                for x_cord in range(self.input_tensor.shape[-1]):
                    self.assertAlmostEqual(temp[0][batch, :, y_cord, x_cord].sum().item(), 1.0, places=5)
    
    def test_apply_standard_activation(self):
        tracker = [False, False, True, True, False, False]
        channels = [None] * self.input_tensor.shape[-3]
        channels = apply_standard_activation(self.input_tensor, self.standard_loss, tracker, channels)
        for i, channel in enumerate(tracker):
            if not channel:
                self.assertTrue(torch.allclose(channels[i], self.standard_loss(torch.select(self.input_tensor, dim=-3, index=i))))

    # def test_mixed_activation(self):
    #     output = mixed_activation(self.input_tensor, self.mutually_exclusive_loss, self.standard_loss, self.mutually_exclusive)
    #     self.assertEqual(output.shape, self.input_tensor.shape)
    #     for sets in self.mutually_exclusive:
    #         for channel in sets:
    #             batch = 0
    #             y_cord = 0
    #             x_cord = 0
    #             self.assertAlmostEqual(output[batch, channel, y_cord, x_cord].sum().item(), 1.0, places=1)

if __name__ == '__main__':
    unittest.main()
