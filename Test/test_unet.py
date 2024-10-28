import unittest
import sys
import os
import torch

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from unet import Unet

class TestUnet(unittest.TestCase):
    
    def setUp(self):
        self.unet = Unet()

    def test_initialization(self):
        # Test if the model initializes correctly with default parameters
        self.assertIsInstance(self.unet, Unet)
        self.assertEqual(len(self.unet.encoder_layers), 24)
        self.assertEqual(len(self.unet.decoder_layers), 23)
        self.assertIsInstance(self.unet.final_conv, torch.nn.Conv2d)
        self.assertIsInstance(self.unet.final_activation, torch.nn.Sigmoid)
        
    def test_validate_parameters(self):
        # Test the validation of parameters
        with self.assertRaises(ValueError):
            Unet(conv_kern_size=2)  # Kernel size must be odd and greater or equal to 3

    def test_calculate_padding(self):
        # Test the padding calculation
        self.assertEqual(self.unet.calculate_padding(3), 1)
        self.assertEqual(self.unet.calculate_padding(5), 2)
    
    def test_encoder_and_decoder_layers(self):
        # Test encoder layers creation
        layers, output_channels = self.unet.create_encoder_layers()
        self.assertEqual(len(layers), 24)
        self.assertEqual(output_channels, 512)
        # Test decoder layers creation
        layers, output_channels = self.unet.create_decoder_layers(output_channels)
        self.assertEqual(len(layers), 23)
        self.assertEqual(output_channels, 32)

    def test_job_6(self):
        model = Unet(
            encoder_pairs=4,
            initial_features=4,
            input_channels=1,
            output_channels=1
        )
        x = torch.randn(5,1,512,512)
        model(x)
        
if __name__ == '__main__':
    unittest.main()