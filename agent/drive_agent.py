import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import models 



class PerceptionEncoder(nn.Mdule): 

    def __init__(self,): 
        super(PerceptionEncoder, self).__init__()

        self.resnet = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x): 

        x1 = self.resnet(x)

        return x1 

         




# class DriveAgent(nn.Module):

#     def __init__(self)