# system
import os

# torch
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
# local
from .layers import DownBlock, Conv, ResnetTransformer
sampling_align_corners = False

from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet

# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 64, 64, 64, 64], }
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [64, 64, 64, 64, 64, 64, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks

class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity, concat_input=True, out_channels=2, encoder=None):
        
        super().__init__()

        self.conv_same=Conv2dSamePadding(2, 3, 1)

        self.encoder = encoder
        self.concat_input = concat_input
        self.up_conv1 = up_conv(2560, 512)
        self.double_conv1 = double_conv(672, 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(336, 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(176, 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(96, 64)

        self.double_conv_input_conv=Conv2dSamePadding(34, 35, 1)
        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(35, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]
    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]
        
    def forward(self, x):
    #def forward(self, x):
        if len(x)==2:
            x = torch.cat(x, 1)
        input_ = x
        print("resUnet forward ", x.shape)
        if x.shape[1]==2:
            x=self.conv_same(x)
        blocks = get_blocks_to_be_concat(self.encoder, x)
        #print("resunet blocks ", blocks)
        _, x = blocks.popitem()
        print("resUnet up_conv1_input ", x.shape)
    
        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            if x.shape[1]==34:
                x=self.double_conv_input_conv(x)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x
      
    
def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = ResUnet(out_channels, 0, 'A', 'kaiming', True, encoder=encoder)
        
    return model
    
class Reg(nn.Module):
    def __init__(self,height,width,in_channels_a,in_channels_b):
        super(Reg, self).__init__()
       #height,width=256,256
        #in_channels_a,in_channels_b=1,1
        init_func = 'kaiming'
        init_to_identity = True

        # paras end------------

        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offset_map = ResUnet(self.in_channels_a, self.in_channels_b, cfg='A', init_func=init_func, init_to_identity=init_to_identity, encoder=get_efficientunet_b7()).to(
            self.device)
        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def forward(self, img_a, img_b, apply_on=None):
        print("reg forward", img_a.shape, img_b.shape)
        deformations = self.offset_map([img_a, img_b])

        return deformations
