import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from libs.DCNv2.dcn_v2 import DCN

# key = deconv layer index, value = feature extractor layer index
# all commented output shapes are for input size of 800x800
SKIP_MAPPING = {
    # 6: 3,  # last relu with 200x200
    3: 6,  # last relu with 100x100
    0: 13,  # last relu with 50x50
}

SKIP_MAPPING_REVERSED = {v: k for k, v in SKIP_MAPPING.items()}


class CenterMobileNetV2(nn.Module):
    def __init__(self, heads, pretrained, freeze_base=False,
                 use_dcn=False, use_skip=False, rotated_boxes=False):
        super(CenterMobileNetV2, self).__init__()

        head_conv = 64

        self.use_skip = use_skip
        self.inplanes = 1280
        self.deconv_with_bias = False
        self.down_ratio = 4
        self.rotated_boxes = rotated_boxes
        mobilenet_v2 = torch.hub.load(
            'pytorch/vision:v0.6.0',
            'mobilenet_v2',
            pretrained=pretrained)
        # skip remove pooling and fc layer from mobilenet_v2
        self.base = mobilenet_v2.features

        if freeze_base:
            for layer in self.base.parameters():
                layer.requires_grad = False

        # if use_dcn:
        #     self.deconv_layers = self._make_deconv_layer(
        #         3,
        #         [256, 128, 64],
        #         [4, 4, 4],
        #         use_dcn,
        #     )
        # else:
        #     self.deconv_layers = self._make_deconv_layer(
        #         3,
        #         [256, 256, 256],
        #         [4, 4, 4],
        #         use_dcn,
        #     )

        self.deconv_layer_channels = [256, 256, 256]
        self.deconv_layers = self._make_deconv_layer(
            3,
            self.deconv_layer_channels,
            [4, 4, 4],
            use_dcn,
        )

        self.skip_convs = {}
        if self.use_skip:
            for deconv_id, fe_id in SKIP_MAPPING.items():
                in_channels = self.base[fe_id].conv[-2].out_channels
                out_channels = self.deconv_layers[deconv_id].out_channels

                self.__setattr__(
                    f"skip_{deconv_id}", nn.Conv2d(
                        in_channels, out_channels, 1, padding=0))

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

    def forward(self, x):
        if self.use_skip:
            skip = {}
            for lid, layer in enumerate(self.base):
                x = layer(x)

                if lid in SKIP_MAPPING.values():
                    skip[SKIP_MAPPING_REVERSED[lid]] = x

            for lid, layer in enumerate(self.deconv_layers):
                x = layer(x)

                if lid in skip:
                    sx = self.__getattr__(f"skip_{lid}")(skip[lid])
                    x = sx + x

        else:
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

    def _make_deconv_layer(self, num_layers, num_filters,
                           num_kernels, use_dcn=False):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            if use_dcn:
                layers.append(DCN(self.inplanes, planes,
                                  kernel_size=(3, 3), stride=1,
                                  padding=1, dilation=1, deformable_groups=1))
                layers.append(nn.BatchNorm2d(planes, momentum=0.1))
                layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes if not use_dcn else planes,
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


def build(num_classes, num_keypoints=0, pretrained=True, freeze_base=False,
          use_dcn=False, use_skip=False, rotated_boxes=False):

    heads = {
        'hm': num_classes,
        'wh': 2 if not rotated_boxes else 3,
        'reg': 2
    }

    if num_keypoints > 0:
        heads['kps'] = num_keypoints * 2

    return CenterMobileNetV2(heads,
                             pretrained=pretrained,
                             freeze_base=freeze_base,
                             use_dcn=use_dcn,
                             use_skip=use_skip,
                             rotated_boxes=rotated_boxes)
