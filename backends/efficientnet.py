import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CenterEfficientNet(nn.Module):
    def __init__(self, variant, heads, pretrained,
                 freeze_base=False, rotated_boxes=False):
        super(CenterEfficientNet, self).__init__()

        head_conv = 64

        self.deconv_with_bias = False
        self.down_ratio = 4
        self.rotated_boxes = rotated_boxes
        self.base = torch.hub.load(
            'lukemelas/EfficientNet-PyTorch',
            f'efficientnet_{variant}',
            pretrained=pretrained)
        self.inplanes = self.base._bn1.num_features
        self.dim_feature_extractor = self.base._bn1.num_features

        if freeze_base:
            for layer in self.base.parameters():
                layer.requires_grad = False

        self.deconv_layer_channels = [256, 256, 256]
        self.deconv_layers = self._make_deconv_layer(
            3,
            self.deconv_layer_channels,
            [4, 4, 4],
        )

        self.heads = heads
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(
                    256,
                    head_conv,
                    kernel_size=3,
                    padding=1,
                    bias=True),
                nn.ReLU(
                    inplace=True),
                nn.Conv2d(
                    head_conv,
                    num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            self.__setattr__(head, fc)

    def forward(self, x, return_features=False, head_only=False):
        f = self.base.extract_features(x)
        if head_only:
            f = f.detach()

        x = self.deconv_layers(f)

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)

        if return_features:
            return z, f

        return z

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters,
                           num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


def build(num_classes, variant='b0', pretrained=True, freeze_base=False,
          rotated_boxes=False):

    if variant not in [f"b{x}" for x in range(0, 9)]:
        raise NotImplementedError(
            f"EffcientNet variant {variant} is not implemented!")

    heads = {
        'hm': num_classes,
        'wh': 2 if not rotated_boxes else 3,
        'reg': 2
    }
    return CenterEfficientNet(variant, heads,
                              pretrained=pretrained,
                              freeze_base=freeze_base,
                              rotated_boxes=rotated_boxes)
