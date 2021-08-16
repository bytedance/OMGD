from .vgg import Vgg16
import torch.nn as nn

class VGGFeature(nn.Module):
    def __init__(self):
        super(VGGFeature, self).__init__()
        self.add_module('vgg', Vgg16())
    def __call__(self,x):
        x = (x.clone()+1.)/2.
        x_vgg = self.vgg(x)
        return x_vgg
