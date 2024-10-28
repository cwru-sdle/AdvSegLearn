import torch
class adversarial_loss(torch.nn.Module):
    def __init__(self):
        super(adversarial_loss,self).__init__()
    def forward(self,disc_output:torch.Tensor):
        shape = disc_output.shape
        eps = 1e-8 
        if len(shape)!= 4:
            raise IndexError('This tensor has more than 3 dimensions, so is incorrectly sectioned')
        if ground_truth:
            return -1 * torch.mean(torch.log(disc_output+eps))
        else:
            return -1 * torch.mean(torch.log(1 - disc_output + eps))