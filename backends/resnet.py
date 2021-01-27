import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

RESNET_MODELS = {
    18: 512,
    34: 512,
    50: 2048,
    101: 2048,
    152: 2048
}


class CenterResNet(nn.Module):
    def __init__(self, num_layers, heads, pretrained,
                 freeze_base=False, rotated_boxes=False):
        super(CenterResNet, self).__init__()

        base_name = f'resnet{num_layers}'
        head_conv = 64

        self.inplanes = RESNET_MODELS[num_layers]
        self.deconv_with_bias = False
        self.down_ratio = 4
        self.rotated_boxes = rotated_boxes
        resnet = torch.hub.load('pytorch/vision:v0.6.0',
                                base_name, pretrained=pretrained)
        # skip remove pooling and fc layer from resnet
        self.base = torch.nn.Sequential(*(list(resnet.children())[:-2]))

        if freeze_base:
            for layer in self.base.parameters():
                layer.requires_grad = False

        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        self.heads = heads
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                          kernel_size=1, stride=1, padding=0)
            )
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.deconv_layers(x)

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
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

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
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


def build(num_layers, num_classes, num_keypoints=0,
          pretrained=True, freeze_base=False, rotated_boxes=False):

    assert num_layers in RESNET_MODELS.keys()

    heads = {
        'hm': num_classes,
        'wh': 2 if not rotated_boxes else 3,
        'reg': 2
    }

    if num_keypoints > 0:
        heads['kps'] = num_keypoints * 2

    return CenterResNet(num_layers, heads,
                        pretrained=pretrained,
                        freeze_base=freeze_base,
                        rotated_boxes=rotated_boxes)
