import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# key = deconv layer index, value = feature extractor layer index
# all commented output shapes are for input size of 512x512
SKIP_MAPPINGS = {
    "b0": {
        5: 4,  # Cx64x64
        2: 10  # Cx32x32
    },
    "b1": {
        5: 7,  # Cx64x64
        2: 15  # Cx32x32
    },
    "b2": {
        5: 7,  # Cx64x64
        2: 15  # Cx32x32
    },
    "b3": {
        5: 7,  # Cx64x64
        2: 17  # Cx32x32
    },
    "b7": {
        5: 17,  # Cx64x64
        2: 37  # Cx32x32
    }
}


SKIP_MAPPINGS_REVERSED = {}

for variant, mapping in SKIP_MAPPINGS.items():
    SKIP_MAPPINGS_REVERSED[variant] = {v: k for k, v in mapping.items()}


class CenterEfficientNet(nn.Module):
    def __init__(
            self, variant, heads, pretrained, freeze_base=False,
            use_skip=False, rotated_boxes=False, use_upsample=False,
            num_head_channels=256, num_deconv_channels=[256, 256, 256]):
        super(CenterEfficientNet, self).__init__()

        assert len(num_deconv_channels) == 3

        head_conv = num_head_channels
        self.use_skip = use_skip
        self.deconv_with_bias = False
        self.down_ratio = 4
        self.variant = variant
        self.rotated_boxes = rotated_boxes
        self.base = torch.hub.load(
            'lukemelas/EfficientNet-PyTorch',
            f'efficientnet_{variant}',
            pretrained=pretrained)
        self.inplanes = self.base._bn1.num_features
        self.use_upsample = use_upsample

        if freeze_base:
            for layer in self.base.parameters():
                layer.requires_grad = False

        self.deconv_layer_channels = num_deconv_channels
        self.deconv_layers = self._make_deconv_layer(
            3,
            self.deconv_layer_channels,
            [4, 4, 4],
        )

        if self.use_skip:
            for i, (deconv_id, fe_id) in enumerate(
                    SKIP_MAPPINGS[self.variant].items()):
                in_channels = self.base._blocks[fe_id]._project_conv.out_channels
                # -2 because conv, bn, relu

                if self.use_upsample:
                    out_channels = self.deconv_layers[deconv_id -
                                                      i *
                                                      1].out_channels
                else:
                    out_channels = self.deconv_layers[deconv_id - 2].out_channels

                skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

                self.__setattr__(
                    f"skip_{deconv_id}", skip)

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
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv,
                    num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            self.__setattr__(head, fc)

    def forward_to_deconv(self, x):
        if self.use_skip:
            skip = {}
            x = self.base._swish(self.base._bn0(self.base._conv_stem(x)))
            # Blocks
            for idx, block in enumerate(self.base._blocks):
                drop_connect_rate = self.base._global_params.drop_connect_rate
                if drop_connect_rate:
                    # scale drop connect_rate
                    drop_connect_rate *= float(idx) / len(self.base._blocks)
                x = block(x, drop_connect_rate=drop_connect_rate)

                if idx in SKIP_MAPPINGS[self.variant].values():
                    skip[SKIP_MAPPINGS_REVERSED[self.variant][idx]] = x

            x = self.base._swish(self.base._bn1(self.base._conv_head(x)))

            for lid, layer in enumerate(self.deconv_layers):
                x = layer(x)

                if lid in skip:
                    sx = self.__getattr__(f"skip_{lid}")(skip[lid])
                    x = sx + x

        else:
            x = self.base.extract_features(x)
            x = self.deconv_layers(x)

        return x

    def forward(self, x):
        x = self.forward_to_deconv(x)

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
            if self.use_upsample:
                layers.append(nn.Upsample(scale_factor=4, mode='bilinear'))
                layers.append(nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=2,
                    padding=padding,
                    bias=self.deconv_with_bias
                ))
            else:
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


def build(num_classes, variant='b0', num_keypoints=0, pretrained=True,
          freeze_base=False, rotated_boxes=False, use_skip=False, **kwargs):

    if variant not in [f"b{x}" for x in range(0, 9)]:
        raise NotImplementedError(
            f"EffcientNet variant {variant} is not implemented!")

    heads = {
        'hm': num_classes,
        'wh': 2 if not rotated_boxes else 3,
        'reg': 2
    }

    if num_keypoints > 0:
        heads['kps'] = num_keypoints * 2

    return CenterEfficientNet(variant, heads,
                              pretrained=pretrained,
                              freeze_base=freeze_base,
                              rotated_boxes=rotated_boxes,
                              use_skip=use_skip, **kwargs)
