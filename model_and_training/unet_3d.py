import torch
from torch import nn
from typing import Union, List, Tuple, Type


class FeatureNoise(nn.Module):
    """ As used in R. Walsh et al. (Oct 2024). Multi-sequence learning for multiple sclerosis lesion segmentation
    in spinal cord MRI. In International Conference on Medical Image Computing and Computer-Assisted Intervention
    (pp. 478-487). Cham: Springer Nature Switzerland."""
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.sigma
        else:
            return x


class SubBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=None, stride=2, padding=None, dropout=0.2,
                 apply_activation=True, norm=nn.InstanceNorm3d, bias=True,
                 feature_noise_sigma: Union[None, float] = None):
        super().__init__()
        if kernel_size is None:
            kernel_size = [3, 3, 3]
        if padding is None:
            padding = [1, 1, 1]
        dropout_block = [nn.Dropout3d(dropout, inplace=True)] if dropout > 0 else []
        if norm == nn.GroupNorm:
            norm_layer = [norm(16, n_channels_out, affine=True)]
        elif norm is not None:
            norm_layer = [norm(n_channels_out, affine=True)]
        else:
            norm_layer = []

        feature_noise = [FeatureNoise(feature_noise_sigma)] if feature_noise_sigma is not None else []
        layers = ([nn.Conv3d(n_channels_in, n_channels_out, kernel_size, stride, padding, bias=bias)] +
                  dropout_block +
                  norm_layer +
                  feature_noise)
        if apply_activation:
            layers += [nn.LeakyReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        return

    def forward(self, inp):
        return self.layers(inp)


class EncoderBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=None, first_stride=None, padding=None,
                 norm: Union[None, Type[nn.Module]] = nn.InstanceNorm3d, dropout=0.0,
                 feature_noise_sigma: Union[None, float] = None):
        super(EncoderBlock, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 3, 3]
        if first_stride is None:
            first_stride = [2, 2, 2]
        if padding is None:
            padding = [1, 1, 1]
        self.layers = nn.Sequential(
            SubBlock(n_channels_in, n_channels_out, kernel_size=kernel_size, stride=first_stride, padding=padding,
                     norm=norm, dropout=dropout, feature_noise_sigma=feature_noise_sigma),
            SubBlock(n_channels_out, n_channels_out, kernel_size=kernel_size, stride=1, padding=padding,
                     norm=norm, dropout=dropout, feature_noise_sigma=feature_noise_sigma))
        return

    def forward(self, inp):
        return self.layers(inp)


class DecoderBlock(nn.Module):
    def __init__(self, n_channels_in, max_num_features, n_conv=2, kernel_size=None, stride=None, padding=None,
                 norm: Union[None, Type[nn.Module]] = nn.InstanceNorm3d, dropout=0.0,
                 feature_noise_sigma: Union[None, float] = None):
        super(DecoderBlock, self).__init__()
        if kernel_size is None:
            kernel_size = [2, 2, 2]
        if stride is None:
            stride = [2, 2, 2]
        if padding is None:
            padding = [1, 1, 1]
        self.up_conv = nn.ConvTranspose3d(min(2 * n_channels_in, max_num_features), n_channels_in, kernel_size=stride,
                                          stride=stride, bias=False)
        layers = [SubBlock(2 * n_channels_in, n_channels_in, kernel_size=kernel_size, stride=1, padding=padding,
                           dropout=dropout, norm=norm, feature_noise_sigma=feature_noise_sigma)]
        for _ in range(n_conv - 1):
            layers.append(SubBlock(n_channels_in, n_channels_in, kernel_size=kernel_size, stride=1, padding=padding,
                                   norm=norm, dropout=dropout, feature_noise_sigma=feature_noise_sigma))

        self.layers = nn.Sequential(*layers)

        return

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat((x, skip), dim=1)
        return self.layers(x)


class ResidualBlock(nn.Module):
    """ Scaled back version of the BasicBlock class from nnU-Net
    https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/
    residual.py#L13
    """
    def __init__(self, n_channels_in, n_channels_out, kernel_size=None, initial_stride=1, padding=None,
                 norm: Union[None, Type[nn.Module]] = nn.InstanceNorm3d, dropout=0.0,
                 feature_noise_sigma: Union[None, float] = None):
        super().__init__()
        if kernel_size is None:
            kernel_size = [3, 3, 3]
        if padding is None:
            padding = [1, 1, 1]
        self.layers = nn.Sequential(
            SubBlock(n_channels_in, n_channels_out, kernel_size=kernel_size, stride=initial_stride, padding=padding,
                     apply_activation=True, norm=norm, dropout=dropout, feature_noise_sigma=feature_noise_sigma),
            SubBlock(n_channels_out, n_channels_out, kernel_size=kernel_size, stride=1, padding=padding,
                     apply_activation=False, norm=norm, dropout=dropout, feature_noise_sigma=feature_noise_sigma))

        has_stride = ((isinstance(initial_stride, int) and initial_stride != 1) or
                      isinstance(initial_stride, (list, tuple)) and any([i != 1 for i in initial_stride]))
        requires_projection = (n_channels_in != n_channels_out)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(nn.AvgPool3d(initial_stride, initial_stride))
            if requires_projection:
                ops.append(SubBlock(
                    n_channels_in, n_channels_out, kernel_size=1, stride=1, padding=0,
                    apply_activation=False, norm=norm, bias=False, dropout=0.0
                ))
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, inp, skip=None):
        if skip is None:
            skip = inp
        skip = self.skip(skip)
        out = self.layers(inp)
        out += skip
        out = nn.LeakyReLU(inplace=True)(out)
        return out


class BaseEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 norm_op: Union[None, Type[nn.Module]] = nn.InstanceNorm3d,
                 return_skips: bool = False,
                 dropout=0.0,
                 feature_noise_sigma: Union[None, float] = None
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, \
            "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_blocks_per_stage) == n_stages, \
            "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, \
            "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, \
            "strides must have as many entries as we have resolution stages (n_stages). " \
            "Important: first entry is recommended to be 1, else we run strided conv directly on the input"

        self.return_skips = return_skips
        self.out_channels = features_per_stage[-1]
        self.stages = None

        self._initialise_layers(input_channels, n_stages, features_per_stage, kernel_sizes, strides, n_blocks_per_stage,
                                norm_op, dropout, feature_noise_sigma=feature_noise_sigma)

    def _initialise_layers(self, input_channels, n_stages, features_per_stage, kernel_sizes, strides, n_blocks_per_stage,
                           norm_op, dropout, feature_noise_sigma):
        # Needs to be implemented in child classes
        raise NotImplementedError

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.return_skips and i < len(self.stages) - 1:
                skips.append(x)
        return x, skips


class PlainEncoder(BaseEncoder):
    def _initialise_layers(self, input_channels, n_stages, features_per_stage, kernel_sizes, strides, n_blocks_per_stage,
                         norm_op, dropout, feature_noise_sigma: Union[None, float] = None):
        stages = []
        for i in range(n_stages):
            n_channels_in = input_channels if i == 0 else features_per_stage[i - 1]
            n_channels_out = features_per_stage[i]

            stage = [SubBlock(n_channels_in, n_channels_out, kernel_size=kernel_sizes[i], stride=strides[i],
                              norm=norm_op, dropout=dropout, feature_noise_sigma=feature_noise_sigma)]

            # The first layer in the first stage is not included in the block count (to match nnU-Net block count)
            n_blocks_remaining = n_blocks_per_stage[i] if i == 0 else n_blocks_per_stage[i] - 1

            stage += [SubBlock(n_channels_out, n_channels_out, kernel_size=kernel_sizes[i], stride=1,
                               norm=norm_op, dropout=dropout, feature_noise_sigma=feature_noise_sigma)
                      for _ in range(n_blocks_remaining)]

            stages.append(nn.Sequential(*stage))

        self.stages = nn.ModuleList(stages)
        return


class ResidualEncoder(BaseEncoder):
    """ Scaled back version of the ResidualEncoder class from nnU-Net
    https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/
    building_blocks/residual_encoders.py#L13
    """
    def _initialise_layers(self, input_channels, n_stages, features_per_stage, kernel_sizes, strides, n_blocks_per_stage,
                         norm_op, dropout, feature_noise_sigma: Union[None, float] = None):
        stages = []
        for i in range(n_stages):
            n_channels_out = features_per_stage[i]

            if i == 0:
                # Initial layer is regular convolution without residual connection
                stage = [SubBlock(input_channels, n_channels_out, kernel_size=kernel_sizes[i], stride=strides[i],
                                  norm=norm_op, dropout=dropout, feature_noise_sigma=feature_noise_sigma)]
                stage += [ResidualBlock(n_channels_out, n_channels_out, kernel_size=kernel_sizes[i], initial_stride=1,
                                        norm=norm_op, dropout=dropout, feature_noise_sigma=feature_noise_sigma)
                          for _ in range(n_blocks_per_stage[0])]

            else:
                stage = [ResidualBlock(features_per_stage[i - 1], n_channels_out, kernel_size=kernel_sizes[i],
                                       initial_stride=strides[i], norm=norm_op, dropout=dropout,
                                       feature_noise_sigma=feature_noise_sigma)]

                stage += [ResidualBlock(n_channels_out, n_channels_out, kernel_size=kernel_sizes[i],
                                        initial_stride=1, norm=norm_op, dropout=dropout,
                                        feature_noise_sigma=feature_noise_sigma)
                          for _ in range(n_blocks_per_stage[i] - 1)]

            stages.append(nn.Sequential(*stage))
        self.stages = nn.ModuleList(stages)
        return


class UNet(nn.Module):
    """ Create UNet by specifying a priori specific conv and pool operations, which can be anisotropic. """

    def __init__(self, conv_kernel_sizes, pool_op_kernel_sizes, n_modalities=1, base_num_channels=32,
                 max_num_features=320, n_classes=2, n_conv_per_stage_encoder: Union[int, List[int], Tuple[int]] = 2,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int]] = 2, deep_supervision=True,
                 residual_encoder=False, norm_op=nn.InstanceNorm3d, dropout=0.0,
                 feature_noise_sigma: Union[None, float] = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        n_channels = [n_modalities, base_num_channels]
        n = len(conv_kernel_sizes)
        for i in range(n - 1):
            n_channels.append(min(n_channels[-1] * 2, max_num_features))

        if isinstance(n_conv_per_stage_encoder, int):
            n_conv_per_stage_encoder = [n_conv_per_stage_encoder] * n
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n-1)

        # Only want padding if we have kernel size > 1
        padding = [[k[0] // 2, k[1] // 2, k[2] // 2] for k in conv_kernel_sizes]

        encoder_class = ResidualEncoder if residual_encoder else PlainEncoder
        self.encoder_and_bottleneck = encoder_class(
            input_channels=n_modalities, n_stages=n, features_per_stage=n_channels[1:], kernel_sizes=conv_kernel_sizes,
            strides=pool_op_kernel_sizes, n_blocks_per_stage=n_conv_per_stage_encoder, norm_op=norm_op,
            return_skips=True, dropout=dropout, feature_noise_sigma=feature_noise_sigma
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(n_channels[n - i], max_num_features=max_num_features, n_conv=n_conv_per_stage_decoder[i-1],
                         kernel_size=conv_kernel_sizes[-(i + 1)], stride=pool_op_kernel_sizes[-i],
                         padding=padding[-(i + 1)], dropout=dropout, norm=norm_op, feature_noise_sigma=feature_noise_sigma)
            for i in range(1, n)
        ])
        nds = n if deep_supervision else 2
        # Note: the range(1,..) here means we don't do deep supervision on the bottleneck - keep for now but just be aware of the difference with previous & subsequent methods
        self.deep_supervision_convs = [nn.Sequential(nn.Conv3d(n_channels[nds - i], n_classes, 1, bias=False))
                                       for i in range(1, nds)]

        self.decoder = nn.ModuleList(self.decoder)
        self.deep_supervision_convs = nn.ModuleList(self.deep_supervision_convs)
        return

    def forward(self, inp):
        res, encoder_outputs = self.encoder_and_bottleneck(inp)
        deep_supervision_segs = []
        for i, layer in enumerate(self.decoder):
            res = layer(res, encoder_outputs[-1 - i])
            if self.deep_supervision and i < len(self.deep_supervision_convs) - 1:
                deep_supervision_segs.append(self.deep_supervision_convs[i](res))

        # Get the final prediction layer for the lesion segmentation
        lesion_output = self.deep_supervision_convs[-1](res)

        if self.deep_supervision:
            return deep_supervision_segs + [lesion_output]
        else:
            return lesion_output

