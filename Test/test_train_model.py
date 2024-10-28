import unittest
import torch
import os
import sys
from unittest.mock import patch

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


from train_model import (
    save_tensor, initialize_optimizer, train_epoch, validate_epoch, plot_losses, train_model
) 
from unet import Unet

class TestTrainingFunctions(unittest.TestCase):
    
    def setUp(self):
        self.model = Unet()
        self.train_dl = [(
            torch.rand(1, 1, 512, 512), 
            torch.rand(1, 1, 512, 512)),
            (torch.rand(1, 1, 512, 512), 
            torch.rand(1, 1, 512, 512))]
        self.val_dl = self.train_dl
        self.optimizer = None
        self.loss_fn = torch.nn.BCELoss()
        self.save_path = ''
        self.accumulation_steps = 1
    
    def test_save_tensor(self):
        tensor = torch.rand(1, 512, 512)
        save_tensor(tensor, 'test.png', self.save_path)
        # Check if the file was saved
        # For unit testing purposes, you might mock plt.savefig
    
    def test_initialize_optimizer(self):
        optimizer = initialize_optimizer(self.model, None, 0.001)
        self.assertIsInstance(optimizer, torch.optim.Adam)
    
    def test_train_epoch(self):
        self.optimizer = initialize_optimizer(self.model, None, 0.001)
        losses = train_epoch(
            self.model, self.train_dl, self.optimizer, self.loss_fn, 
            self.accumulation_steps, self.save_path, None, 0, 1
        )
        self.assertGreater(len(losses), 0)
    
    def test_validate_epoch(self):
        losses = validate_epoch(self.model, self.val_dl, self.loss_fn)
        self.assertGreater(len(losses), 0)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_losses(self, mock_savefig):
        plot_losses([1, 2, 3], [1, 2, 3], 3, self.save_path)
        mock_savefig.assert_called_once_with(self.save_path + '/loss_figure.png')

if __name__ == '__main__':
    unittest.main()
