# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.sampler import FullSampler

class CosimLoss (nn.Module):
    """ Try to make the repeatability repeatable from one image to the other.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'cosim{N}'
        self.patches = nn.Unfold(N, padding=0, stride=N//2)

    def extract_patches(self, sal):
        patches = self.patches(sal).transpose(1,2) # flatten
        patches = F.normalize(patches, p=2, dim=2) # norm
        return patches
        
    def forward(self, repeatability, aflow, **kw):
        B,two,H,W = aflow.shape
        assert two == 2

        # normalize
        sali1, sali2 = repeatability
        grid = FullSampler._aflow_to_grid(aflow)
        sali2 = F.grid_sample(sali2, grid, mode='bilinear', padding_mode='border')

        patches1 = self.extract_patches(sali1)
        patches2 = self.extract_patches(sali2)
        cosim = (patches1 * patches2).sum(dim=2)
        return 1 - cosim.mean()


class PeakyLoss (nn.Module):
    """ Try to make the repeatability locally peaky.

    Mechanism: we maximize, for each pixel, the difference between the local mean
               and the local max.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'peaky{N}'
        assert N % 2 == 0, 'N must be pair'
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)

    def forward_one(self, sali):
        sali = self.preproc(sali) # remove super high frequency
        return 1 - (self.maxpool(sali) - self.avgpool(sali)).mean()

    def forward(self, repeatability, **kw):
        sali1, sali2 = repeatability
        return (self.forward_one(sali1) + self.forward_one(sali2)) /2

class DenseGuidedLoss(nn.Module):
    """ Exploit gradient to guide the learning of detection map.
    Areas with large gradient should be paid more attention.

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.name = 'gradient_guided'
        
    def forward(self, repeatability, image_gradient, **kw):
        
        repeatability1, repeatability2 = repeatability
        gradient1, gradient2 = image_gradient
        
        B, C, H, W = gradient1.shape
        gradient1 = gradient1.squeeze().reshape(B, -1)
        gradient2 = gradient2.squeeze().reshape(B, -1)
        repeatability1 = repeatability1.squeeze().reshape(B, -1)
        repeatability2 = repeatability2.squeeze().reshape(B, -1)
        

        loss = torch.mean(torch.norm(repeatability1 - gradient1, dim=-1) + torch.norm(repeatability2 - gradient2, dim=-1))
        return loss



