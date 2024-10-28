import torch
import unittest
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from train_GAN import (
    train_GAN, metrics_list,set_generator_optimizer,set_discriminator_optimizer,segmented_training_epoch,validation_epoch,unsegmented_training_epoch,save_results
)
from discriminator_loss import discriminator_loss
from semi_supervised_loss import semi_supervised_loss
from unet import Unet
class TestTrainGAN(unittest.TestCase):
    def setUp(self):
        self.seg_model  = Unet(
    input_channels = 1,
    output_channels=1,
    encoder_pairs=4,
    initial_features=64,
    features_expanded=2,
)
        self.disc_model = Unet()
        self.seg_gen_loss = torch.nn.BCELoss()
        self.seg_disc_loss = discriminator_loss()
        self.unseg_gen_loss = semi_supervised_loss()
        
        self.seg_train_dl = [(
            torch.rand(1, 1, 512, 512), 
            torch.rand(1, 1, 512, 512)),
            (torch.rand(1, 1, 512, 512), 
            torch.rand(1, 1, 512, 512))]
        self.seg_val_dl = self.seg_train_dl
        self.unseg_train_dl = self.seg_train_dl

    def test_metrics_list(self):
        input_list = [1.0, 2.0, 3.0, 4.0]
        mean, p25, p50, p75 = metrics_list(input_list)
        self.assertEqual(mean, 2.5)
        self.assertEqual(p25, 1.75)
        self.assertEqual(p50, 2.5)
        self.assertEqual(p75, 3.25)

    def test_set_generator_optimizer(self):
        optimizer = set_generator_optimizer(None, self.seg_model, 0.001)
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertTrue(len(optimizer.param_groups[0]['params']) > 0)
    
    def test_set_discriminator_optimizer(self):
        optimizer = set_discriminator_optimizer(None, self.disc_model, 0.001)
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertTrue(len(optimizer.param_groups[0]['params']) > 0)
    
    def test_train_GAN(self):
        train_GAN(
            seg_disc_loss=self.seg_disc_loss,
            seg_gen_loss=self.seg_gen_loss,
            unseg_gen_loss=self.unseg_gen_loss,
            seg_model=self.seg_model,
            disc_model=self.disc_model,
            seg_train_dl=self.seg_train_dl,
            seg_val_dl=self.seg_val_dl,
            unseg_train_dl=self.unseg_train_dl,
            epochs=1,
            accumulation_steps=1,
            learning_rate=0.0001
        )

        # If no errors, the test is successful
    def test_segmented_training_epoch(self):
        gen_optimizer = set_generator_optimizer(None, self.seg_model, 0.001)
        disc_optimizer = set_discriminator_optimizer(None, self.disc_model, 0.001)
        train_gen_loss_metrics, synthetic_loss_metrics, ground_truth_loss_metrics = segmented_training_epoch(
            seg_train_dl=self.seg_train_dl,
            seg_model=self.seg_model,
            seg_gen_loss=self.seg_gen_loss,
            accumulation_steps=1,
            gen_optimizer=gen_optimizer,
            seg_disc_loss=self.seg_disc_loss,
            disc_model=self.disc_model,
            disc_optimizer=disc_optimizer,
            train_gen_loss_metrics=[],
            synthetic_loss_metrics=[],
            ground_truth_loss_metrics=[]
        )
        self.assertIsInstance(train_gen_loss_metrics, list)
        self.assertIsInstance(synthetic_loss_metrics, list)
        self.assertIsInstance(ground_truth_loss_metrics, list)
    
    def test_validation_epoch(self):
        val_gen_loss_metrics = []
        val_gen_loss_metrics = validation_epoch(
            seg_val_dl=self.seg_val_dl,
            seg_model=self.seg_model,
            seg_gen_loss=self.seg_gen_loss,
            val_gen_loss_metrics=val_gen_loss_metrics
        )
        self.assertIsInstance(val_gen_loss_metrics, list)

    def test_unsegmented_training_epoch(self):
        gen_optimizer = set_generator_optimizer(None, self.seg_model, 0.001)
        semisupervsied_loss_metrics = unsegmented_training_epoch(
            unseg_train_dl=self.unseg_train_dl,
            seg_model=self.seg_model,
            disc_model=self.disc_model,
            unseg_gen_loss=self.unseg_gen_loss,
            semisupervsied_loss_metrics=[],
            accumulation_steps=1,
            gen_optimizer=gen_optimizer
        )
        self.assertIsInstance(semisupervsied_loss_metrics, list)


    def test_save_results(self):
        save_path = 'test_results.png'
        epochs = 2
        loss_metrics = [
            [(1.0, 0.5, 1.5, 2.0), (2.0, 1.5, 2.5, 3.0)],
            [(2.0, 1.5, 2.5, 3.0), (3.0, 2.5, 3.5, 4.0)],
            [(3.0, 2.5, 3.5, 4.0), (4.0, 3.5, 4.5, 5.0)],
            [(4.0, 3.5, 4.5, 5.0), (5.0, 4.5, 5.5, 6.0)],
            [(5.0, 4.5, 5.5, 6.0), (6.0, 5.5, 6.5, 7.0)],
        ]
        save_results(save_path, epochs, loss_metrics)
        import os
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)
if __name__ == '__main__':
    unittest.main()