import torch.nn as nn
from typing import Tuple

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.network_initialization import InitWeights_He


def get_nnunet_3d(
    in_channels: int = 3,
    out_channels: int = 1,
    n_stages: int = 5,
    features_per_stage: Tuple[int, ...] = (32, 64, 128, 256, 320),
    deep_supervision: bool = False,
) -> PlainConvUNet:
    """Build a 3D nnU-Net style PlainConvUNet."""
    if len(features_per_stage) != n_stages:
        raise ValueError("features_per_stage length must match n_stages")

    kernel_sizes = [[3, 3, 3]] * n_stages
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    n_conv_per_stage = [2] * n_stages
    n_conv_per_stage_decoder = [2] * (n_stages - 1)

    model = PlainConvUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=out_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    model.apply(InitWeights_He(1e-2))
    return model


def NNUNet3D(
    in_channels: int = 3,
    out_channels: int = 1,
    deep_supervision: bool = False,
) -> PlainConvUNet:
    return get_nnunet_3d(
        in_channels=in_channels,
        out_channels=out_channels,
        deep_supervision=deep_supervision,
    )
