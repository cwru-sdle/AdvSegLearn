import torch
import unittest
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from semi_supervised_loss import semi_supervised_loss

class TestSemiSupervisedLoss(unittest.TestCase):
    def setUp(self):
        self.THRESHOLD = .3 #THESHOLD should work within range(0,1)
        self.loss_fn = semi_supervised_loss(self.THRESHOLD,[{1,2},{4,5}])
        self.SHAPE = torch.zeros(5,6,9,9)
    
    def randn01(self,shape:torch.Tensor):
        randn_tensor = torch.randn_like(shape)
        return (randn_tensor - randn_tensor.min()) / (randn_tensor.max() - randn_tensor.min())
    def test_indicator_below_scalar(self):
        x = torch.zeros(1).requires_grad_(requires_grad=True).to('cuda:0')
        x = self.loss_fn.indicator_function(x)
        x.backward()
        self.assertTrue(x.grad_fn!=None)
        x = x.item()
        self.assertAlmostEqual(x,0.0)

    def test_indicator_above_scalar(self):
        x = torch.ones(1).requires_grad_(requires_grad=True).to('cuda:0')
        x = self.loss_fn.indicator_function(x)
        x.backward()
        self.assertTrue(x.grad_fn!=None)
        x = x.item()
        self.assertAlmostEqual(x,1.0)

        #Test at threshold, single value- it can't both be differentiable and work at the threshold
        # x = torch.zeros(1) + THRESHOLD
        # x.requires_grad_(requires_grad=True)
        # x = loss_fn.indicator_function(x)
        # x.backward()
        # self.assertTrue(x.grad_fn!=None)seg_temp
        # x = x.item()
        # self.assertAlmostEqual(x,0.0)


    def test_indicator_below_tensor(self):
        for device in ['cuda:0','cpu']:
            x = torch.zeros_like(self.SHAPE).requires_grad_(requires_grad=True).to(device)
            x = self.loss_fn.indicator_function(x)
            self.assertAlmostEqual(x.shape,self.SHAPE.shape)
            x = torch.sum(x)
            x.backward()
            self.assertTrue(x.grad_fn!=None)
            self.assertTrue(not torch.isnan(x))

    def test_forward_grad(self):
        for device in ['cuda:0','cpu']:
            x = torch.randn_like(self.SHAPE).requires_grad_(requires_grad=True).to(device)
            y = torch.randn_like(self.SHAPE).requires_grad_(requires_grad=True).to(device)
            z = self.loss_fn(x,y)
            z.backward()
            self.assertTrue(z.grad_fn!=None)
    
    def test_seperate_non_exclusive(self):
        for device in ['cuda:0','cpu']:
            x = torch.randn_like(self.SHAPE).requires_grad_(requires_grad=True).to(device)
            tensor_list, position = self.loss_fn.seperate_tensor(x)
            for j, i in enumerate(position):
                if(len(i)==1 and i[0]==0):
                    y = tensor_list[j][0]
            self.assertTrue(torch.equal(torch.select(x,-3,0),y))
            y = torch.sum(y)
            self.assertTrue(not torch.isnan(y))


    def test_forward_positive(self):
        for device in ['cuda:0','cpu']:
            disc_output = torch.zeros_like(self.SHAPE).to(device)
            seg_output = self.randn01(self.SHAPE).to(device)
            z = self.loss_fn(disc_output,seg_output)
            self.assertTrue(z.item()==0)


    def test_forward_basic(self):
        for device in ['cuda:0','cpu']:
            disc_output = self.randn01(self.SHAPE).to(device)
            seg_output = self.randn01(self.SHAPE).to(device)
            z = self.loss_fn(disc_output,seg_output)
            self.assertTrue(type(z.item())==float)
            self.assertTrue(z.item()>=0)

#Make tests that make sure that device compatabilites make all of this workable
if __name__ == '__main__':
    unittest.main()