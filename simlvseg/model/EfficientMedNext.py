from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class MultiDilationDepthwiseConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        strides: Sequence[int] = (1, 1, 1),
        dw_parallel: bool = True,
    ):
        super().__init__()
        self.dw_parallel = dw_parallel
        dilations = [(kernel - 1) // 2 if kernel > 1 else 1 for kernel in kernel_sizes]
        modified_kernel_sizes = [3 if kernel > 1 else 1 for kernel in kernel_sizes]

        self.dwconvs = nn.ModuleList(
            [
                conv(
                    in_channels,
                    in_channels,
                    kernel_size=modified_kernel_sizes[i],
                    stride=strides[i],
                    padding=kernel_sizes[i] // 2,
                    dilation=dilations[i],
                    groups=in_channels,
                    bias=False,
                )
                for i in range(len(modified_kernel_sizes))
            ]
        )

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return torch.cat(outputs, dim=1)


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=None):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        reduce_dims = tuple(range(2, x.ndim))
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        param_shape = [1, -1] + [1] * len(reduce_dims)
        return self.weight.view(*param_shape) * x + self.bias.view(*param_shape)


class EfficientMedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        strides: Sequence[int] = (1, 1, 1),
        do_res: bool = True,
        norm_type: str = "group",
        dim: str = "3d",
        conv=None,
        grn: bool = False,
    ):
        super().__init__()

        self.do_res = do_res
        self.in_channels = in_channels
        self.out_channels = out_channels
        exp_r = len(kernel_sizes)
        assert dim in ["2d", "3d"]
        self.dim = dim
        if conv is None:
            conv = nn.Conv2d if self.dim == "2d" else nn.Conv3d

        self.conv1 = MultiDilationDepthwiseConv3D(
            in_channels,
            conv,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dw_parallel=True,
        )

        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=exp_r * in_channels)
        elif norm_type == "layer":
            self.norm = LayerNorm(normalized_shape=exp_r * in_channels, data_format="channels_first")
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.act = nn.GELU()
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.do_res and (self.in_channels != self.out_channels):
            self.res_conv = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.grn = grn
        if grn:
            shape = (1, exp_r * in_channels, 1, 1, 1) if dim == "3d" else (1, exp_r * in_channels, 1, 1)
            self.grn_beta = nn.Parameter(torch.zeros(shape), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, x, dummy_tensor=None):
        x1 = self.conv1(x)
        x1 = self.act(self.norm(x1))
        if self.grn:
            spatial_dims = (-3, -2, -1) if self.dim == "3d" else (-2, -1)
            gx = torch.norm(x1, p=2, dim=spatial_dims, keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            if self.in_channels != self.out_channels:
                x = self.res_conv(x)
            x1 = x + x1
        return x1


class EfficientMedNeXtDownBlock(EfficientMedNeXtBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        strides: Sequence[int] = (2, 1, 1),
        do_res: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        conv = nn.Conv2d if dim == "2d" else nn.Conv3d
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_sizes,
            strides=strides,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class EfficientMedNeXtUpBlock(EfficientMedNeXtBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        strides: Sequence[int] = (2, 1, 1),
        do_res: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        self.resample_do_res = do_res
        self.dim = dim
        conv = nn.ConvTranspose2d if dim == "2d" else nn.ConvTranspose3d
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_sizes=kernel_sizes,
            strides=strides,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            conv=conv,
            grn=grn,
        )
        if do_res:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        pad = (1, 0, 1, 0) if self.dim == "2d" else (1, 0, 1, 0, 1, 0)
        x1 = F.pad(x1, pad)
        if self.resample_do_res:
            x1 = x1 + F.pad(self.res_conv(x), pad)
        return x1


class OutBlock(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dim: str, stride: int = 1):
        super().__init__()
        conv = nn.ConvTranspose2d if dim == "2d" else nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1, stride=stride)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class EfficientMedNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        kernel_sizes: Optional[Sequence[int]] = (1, 3, 5),
        strides: Sequence[int] = (1, 1, 1),
        enc_kernel_sizes: Sequence[int] = (1, 3, 5),
        dec_kernel_sizes: Sequence[int] = (1, 3, 5),
        uniform_dec_channels: Optional[int] = None,
        deep_supervision: bool = True,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: Optional[str] = None,
        block_counts: List[int] = None,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
        mode: str = "train",
    ):
        super().__init__()

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        if len(block_counts) != 9:
            raise ValueError("block_counts must contain 9 integers")
        if checkpoint_style not in [None, "outside_block"]:
            raise ValueError("checkpoint_style must be None or 'outside_block'")
        assert dim in ["2d", "3d"]

        self.do_ds = deep_supervision
        self.outside_block_checkpointing = checkpoint_style == "outside_block"
        if kernel_sizes is not None:
            enc_kernel_sizes = kernel_sizes
            dec_kernel_sizes = kernel_sizes
        up_down_strides = [s * 2 for s in strides]

        conv = nn.Conv2d if dim == "2d" else nn.Conv3d
        num_channels = [n_channels, n_channels * 2, n_channels * 4, n_channels * 8, n_channels * 16]
        dec_num_channels = list(num_channels) if uniform_dec_channels is None else [uniform_dec_channels] * len(num_channels)

        self.stem = conv(in_channels, num_channels[0], kernel_size=1)
        self.enc_block_0 = self._make_blocks(num_channels[0], num_channels[0], block_counts[0], enc_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.down_0 = EfficientMedNeXtDownBlock(num_channels[0], num_channels[1], kernel_sizes=enc_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim)
        self.enc_block_1 = self._make_blocks(num_channels[1], num_channels[1], block_counts[1], enc_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.down_1 = EfficientMedNeXtDownBlock(num_channels[1], num_channels[2], kernel_sizes=enc_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)
        self.enc_block_2 = self._make_blocks(num_channels[2], num_channels[2], block_counts[2], enc_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.down_2 = EfficientMedNeXtDownBlock(num_channels[2], num_channels[3], kernel_sizes=enc_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)
        self.enc_block_3 = self._make_blocks(num_channels[3], num_channels[3], block_counts[3], enc_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.down_3 = EfficientMedNeXtDownBlock(num_channels[3], num_channels[4], kernel_sizes=enc_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)

        self.crrb4 = EfficientMedNeXtBlock(num_channels[4], dec_num_channels[4], kernel_sizes=dec_kernel_sizes, strides=strides, do_res=do_res, norm_type=norm_type, dim=dim, grn=grn)
        self.bottleneck = self._make_blocks(dec_num_channels[4], dec_num_channels[4], block_counts[4], dec_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.crrb3 = EfficientMedNeXtBlock(num_channels[3], dec_num_channels[3], kernel_sizes=dec_kernel_sizes, strides=strides, do_res=do_res, norm_type=norm_type, dim=dim, grn=grn)
        self.crrb2 = EfficientMedNeXtBlock(num_channels[2], dec_num_channels[2], kernel_sizes=dec_kernel_sizes, strides=strides, do_res=do_res, norm_type=norm_type, dim=dim, grn=grn)
        self.crrb1 = EfficientMedNeXtBlock(num_channels[1], dec_num_channels[1], kernel_sizes=dec_kernel_sizes, strides=strides, do_res=do_res, norm_type=norm_type, dim=dim, grn=grn)
        self.crrb0 = EfficientMedNeXtBlock(num_channels[0], dec_num_channels[0], kernel_sizes=dec_kernel_sizes, strides=strides, do_res=do_res, norm_type=norm_type, dim=dim, grn=grn)

        self.up_3 = EfficientMedNeXtUpBlock(dec_num_channels[4], dec_num_channels[3], kernel_sizes=dec_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)
        self.dec_block_3 = self._make_blocks(dec_num_channels[3], dec_num_channels[3], block_counts[5], dec_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.up_2 = EfficientMedNeXtUpBlock(dec_num_channels[3], dec_num_channels[2], kernel_sizes=dec_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)
        self.dec_block_2 = self._make_blocks(dec_num_channels[2], dec_num_channels[2], block_counts[6], dec_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.up_1 = EfficientMedNeXtUpBlock(dec_num_channels[2], dec_num_channels[1], kernel_sizes=dec_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)
        self.dec_block_1 = self._make_blocks(dec_num_channels[1], dec_num_channels[1], block_counts[7], dec_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.up_0 = EfficientMedNeXtUpBlock(dec_num_channels[1], dec_num_channels[0], kernel_sizes=dec_kernel_sizes, strides=up_down_strides, do_res=do_res_up_down, norm_type=norm_type, dim=dim, grn=grn)
        self.dec_block_0 = self._make_blocks(dec_num_channels[0], dec_num_channels[0], block_counts[8], dec_kernel_sizes, strides, do_res, norm_type, dim, grn)
        self.out_0 = OutBlock(dec_num_channels[0], n_classes, dim=dim)

        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(dec_num_channels[1], n_classes, dim=dim)
            self.out_2 = OutBlock(dec_num_channels[2], n_classes, dim=dim)
            self.out_3 = OutBlock(dec_num_channels[3], n_classes, dim=dim)
            self.out_4 = OutBlock(dec_num_channels[4], n_classes, dim=dim)

        self.block_counts = block_counts

    @staticmethod
    def _make_blocks(in_channels, out_channels, count, kernel_sizes, strides, do_res, norm_type, dim, grn):
        return nn.Sequential(
            *[
                EfficientMedNeXtBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(count)
            ]
        )

    def iterative_checkpoint(self, sequential_block, x):
        for layer in sequential_block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor, use_reentrant=False)
        return x

    def _checkpoint(self, module, x):
        return checkpoint.checkpoint(module, x, self.dummy_tensor, use_reentrant=False)

    def forward(self, x, mode: str = "test"):
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = self._checkpoint(self.down_0, x_res_0)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = self._checkpoint(self.down_1, x_res_1)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = self._checkpoint(self.down_2, x_res_2)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = self._checkpoint(self.down_3, x_res_3)

            x_res_0 = self._checkpoint(self.crrb0, x_res_0)
            x_res_1 = self._checkpoint(self.crrb1, x_res_1)
            x_res_2 = self._checkpoint(self.crrb2, x_res_2)
            x_res_3 = self._checkpoint(self.crrb3, x_res_3)
            x = self._checkpoint(self.crrb4, x)
            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = self._checkpoint(self.out_4, x)

            x = self.iterative_checkpoint(self.dec_block_3, x_res_3 + self._checkpoint(self.up_3, x))
            if self.do_ds:
                x_ds_3 = self._checkpoint(self.out_3, x)
            x = self.iterative_checkpoint(self.dec_block_2, x_res_2 + self._checkpoint(self.up_2, x))
            if self.do_ds:
                x_ds_2 = self._checkpoint(self.out_2, x)
            x = self.iterative_checkpoint(self.dec_block_1, x_res_1 + self._checkpoint(self.up_1, x))
            if self.do_ds:
                x_ds_1 = self._checkpoint(self.out_1, x)
            x = self.iterative_checkpoint(self.dec_block_0, x_res_0 + self._checkpoint(self.up_0, x))
            x = self._checkpoint(self.out_0, x)
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x_res_0 = self.crrb0(x_res_0)
            x_res_1 = self.crrb1(x_res_1)
            x_res_2 = self.crrb2(x_res_2)
            x_res_3 = self.crrb3(x_res_3)
            x = self.crrb4(x)
            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x = self.dec_block_3(x_res_3 + self.up_3(x))
            if self.do_ds:
                x_ds_3 = self.out_3(x)
            x = self.dec_block_2(x_res_2 + self.up_2(x))
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            x = self.dec_block_1(x_res_1 + self.up_1(x))
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            x = self.dec_block_0(x_res_0 + self.up_0(x))
            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x


EfficientMedNeXt_L = EfficientMedNeXt


def create_efficient_mednext_tiny(
    num_input_channels,
    num_classes,
    n_channels=32,
    kernel_sizes=(1, 3, 5),
    strides=(1, 1, 1),
    uniform_dec_channels=None,
    ds=False,
    mode="train",
):
    return EfficientMedNeXt(
        in_channels=num_input_channels,
        n_channels=n_channels,
        n_classes=num_classes,
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        checkpoint_style="outside_block",
        mode=mode,
    )


def create_efficient_mednext_small(
    num_input_channels,
    num_classes,
    n_channels=32,
    kernel_sizes=(1, 3, 5),
    strides=(1, 1, 1),
    uniform_dec_channels=None,
    ds=False,
    mode="train",
):
    return EfficientMedNeXt(
        in_channels=num_input_channels,
        n_channels=n_channels,
        n_classes=num_classes,
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style="outside_block",
        mode=mode,
    )


def create_efficient_mednext_medium(
    num_input_channels,
    num_classes,
    n_channels=32,
    kernel_sizes=(1, 3, 5),
    strides=(1, 1, 1),
    uniform_dec_channels=None,
    ds=False,
    mode="train",
):
    return EfficientMedNeXt(
        in_channels=num_input_channels,
        n_channels=n_channels,
        n_classes=num_classes,
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style="outside_block",
        mode=mode,
    )


def create_efficient_mednext_large(
    num_input_channels,
    num_classes,
    n_channels=32,
    kernel_sizes=(1, 3, 5),
    strides=(1, 1, 1),
    uniform_dec_channels=None,
    ds=False,
    mode="train",
):
    return EfficientMedNeXt_L(
        in_channels=num_input_channels,
        n_channels=n_channels,
        n_classes=num_classes,
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style="outside_block",
        mode=mode,
    )


def create_efficient_mednext(
    num_input_channels,
    num_classes,
    model_id="T",
    n_channels=32,
    kernel_sizes=(1, 3, 5),
    strides=(1, 1, 1),
    uniform_dec_channels=None,
    deep_supervision=False,
    mode="train",
):
    model_dict = {
        "T": create_efficient_mednext_tiny,
        "S": create_efficient_mednext_small,
        "M": create_efficient_mednext_medium,
        "L": create_efficient_mednext_large,
    }
    return model_dict[model_id.upper()](
        num_input_channels,
        num_classes,
        n_channels,
        kernel_sizes,
        strides,
        uniform_dec_channels,
        deep_supervision,
        mode=mode,
    )
