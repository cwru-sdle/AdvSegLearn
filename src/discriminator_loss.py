import torch
class discriminator_loss(torch.nn.Module):
    def __init__(self):
        super(discriminator_loss,self).__init__()
    def forward(self,disc_output:torch.Tensor,ground_truth:bool):
        shape = disc_output.shape
        eps = 1e-8 
        if ground_truth:
            return -1 * torch.mean(torch.log(disc_output+eps))
        else:
            return -1 * torch.mean(torch.log(1 - disc_output + eps))