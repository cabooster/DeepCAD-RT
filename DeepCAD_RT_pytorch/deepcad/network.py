from .model_3DUnet import UNet3D
import torch.nn as nn

class Network_3D_Unet(nn.Module):
    def __init__(self, UNet_type = '3DUNet', in_channels=1, out_channels=1, f_maps=64, final_sigmoid = True):
        super(Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid

        if UNet_type == '3DUNet':
            self.Generator = UNet3D( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps, 
                                     final_sigmoid = final_sigmoid)

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x
