# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import torch.nn as nn
import torch, einops
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F

import numpy as np
from timm.models.layers import trunc_normal_
import math


# from torchsummary import summary
# from utils.image_visualizer import ImageVisualizer

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class Point:
    def __init__(self, x=0, y=0):
        self.x = x  # X坐标
        self.y = y  # Y坐标


class Hilbert:
    def __init__(self):
        self.hilbert_maps = {}
        for n in [64, 32, 16, 8, 4]:
            self.hilbert_maps[n] = self.precompute_hilbert_map(n)

    def precompute_hilbert_map(self, n):
        hilbert_map = []
        for d in range(n * n):
            pt = Point()
            self.d2xy(n, d, pt)
            hilbert_map.append([pt.x, pt.y])
        return torch.tensor(hilbert_map, dtype=torch.long)  # 存储为张量

    def rot(self, n, pt, rx, ry):
        if ry == 0:
            if rx == 1:
                pt.x = n - 1 - pt.x
                pt.y = n - 1 - pt.y

            # Swap x and y
            pt.x, pt.y = pt.y, pt.x

    # Hilbert代码到XY坐标
    def d2xy(self, n, d, pt):
        pt.x, pt.y = 0, 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            self.rot(s, pt, rx, ry)
            pt.x += s * rx
            pt.y += s * ry
            t //= 4
            s *= 2

    # XY坐标到Hilbert代码转换
    def xy2d(self, n, pt):
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if (pt.x & s) > 0 else 0
            ry = 1 if (pt.y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            self.rot(s, pt, rx, ry)
            s //= 2
        return d


class MooreCurve:
    def __init__(self):
        self.mooreCurveMaps = {}
        # 预计算常见尺寸的映射表
        for side_length in [128, 64, 32, 16, 8, 4]:
            self.mooreCurveMaps[side_length] = self.precompute_moore_curve_map(side_length)

    def precompute_moore_curve_map(self, side_length):
        """预计算指定边长的摩尔曲线映射表"""
        # 计算阶数（根据边长推导）
        n = int(math.log2(side_length))

        # 生成坐标序列
        points = self.moore_curve_order_to_coords(n)

        # 计算坐标范围
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)

        curve_map = []
        # 填充遍历顺序
        for step, (x, y) in enumerate(points):
            curve_map.append((y - min_y, x - min_x))

        return torch.tensor(curve_map, dtype=torch.long)

    def generate_moore_curve_string(self, n):
        """生成n阶L-system字符串 """
        axiom = 'LFL+F+LFL'
        rules = {
            'L': '-RF+LFL+FR-',
            'R': '+LF-RFR-FL+',
        }
        current = axiom
        for _ in range(n - 1):  # 替换次数为n-1次
            new_str = []
            for char in current:
                new_str.append(rules.get(char, char))  # 应用规则或保留原字符
            current = ''.join(new_str)
        return current

    def moore_curve_order_to_coords(self, n):
        """解析L-system字符串生成坐标序列"""
        string = self.generate_moore_curve_string(n)

        # 初始化状态
        x, y = 0, 0
        direction = 0  # 0:东 1:北 2:西 3:南
        points = [(x, y)]  # 包含起点

        # 解析指令
        for char in string:
            if char == 'F':
                # 根据当前方向移动
                if direction == 0:
                    x += 1
                elif direction == 1:
                    y += 1
                elif direction == 2:
                    x -= 1
                elif direction == 3:
                    y -= 1
                points.append((x, y))
            elif char == '+':
                direction = (direction - 1) % 4  # 右转
            elif char == '-':
                direction = (direction + 1) % 4  # 左转

        return points

    @staticmethod
    def coords_to_2d_array(points):
        """将坐标序列转换为二维数组（辅助函数）"""
        if not points:
            return []

        # 计算坐标范围
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # 初始化数组
        rows = max_y - min_y + 1
        cols = max_x - min_x + 1
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        # 填充遍历顺序
        for step, (x, y) in enumerate(points):
            grid[y - min_y][x - min_x] = step + 1

        return grid


class DiagonalMap:
    def __init__(self):
        self.diagonalMaps = {}
        # 预计算常见尺寸的映射表
        for side_length in [128, 64, 32, 16, 8, 4]:
            self.diagonalMaps[side_length] = self.precompute_diagonal_order(side_length)

    def precompute_diagonal_order(self, side_length):
        """预计算坐标映射关系，返回一维坐标到二维坐标的列表"""
        H = side_length
        W = side_length
        coord_map = []
        for s in range(H + W - 1):
            start_row = max(0, s - (W - 1))
            end_row = min(H - 1, s)
            for row in range(end_row, start_row - 1, -1):  # 逆序保证右上到左下
                coord = (row, s - row)  # 存储二维坐标
                coord_map.append(coord)
        return torch.tensor(coord_map, dtype=torch.long)

class ZCurveMapper:
    def __init__(self):
        # 定义需要预计算的边长列表
        self.target_sizes = [4, 8, 16, 32, 64]

        # 存储映射表的字典:
        # Key = 边长 (int), Value = List[(x, y)]
        self.maps = {}

        # 初始化时直接执行预计算
        self._precompute_all()

    def _z_index_to_xy(self, z_index):
        """
        核心位运算逻辑：
        将 Z-index 解码为 (x, y)。
        x = 偶数位组合, y = 奇数位组合。
        """
        # --- 重新实现更稳健的循环逻辑 ---
        x = 0
        y = 0
        for i in range(16):  # 遍历16位足够支持到 256x256
            bit = (z_index >> i) & 1
            if i % 2 == 0:  # 偶数位 -> y
                y |= (bit << (i // 2))
            else:  # 奇数位 -> x
                x |= (bit << (i // 2))
        return x, y

    def _precompute_all(self):
        """内部方法：生成所有指定尺寸的映射表"""

        for size in self.target_sizes:
            total_points = size * size
            # 使用列表作为查找表，Index即为列表下标
            lookup_table = [None] * total_points

            for z in range(total_points):
                lookup_table[z] = self._z_index_to_xy(z)

            self.maps[size] = torch.tensor(lookup_table, dtype=torch.long)

    def get_coord(self, size, z_index):
        """
        获取坐标的主方法
        :param size: 图片边长 (4, 8, 16, 32, 64)
        :param z_index: 一维索引
        :return: (x, y) 元组
        """
        if size not in self.maps:
            raise ValueError(f"Size {size} 未预计算。可用尺寸: {self.target_sizes}")

        # 边界检查
        if z_index < 0 or z_index >= len(self.maps[size]):
            raise IndexError(f"Index {z_index} 超出尺寸 {size}x{size} 的范围")

        return self.maps[size][z_index]

    def print_grid_preview(self, size):
        """工具方法：打印网格预览以验证形状"""
        if size not in self.maps:
            return

        print(f"\n--- Preview Grid: {size}x{size} ---")
        grid = [[0] * size for _ in range(size)]
        table = self.maps[size]

        # 反向填充用于显示：知道 index -> 填入 (x,y)
        for z, (x, y) in enumerate(table):
            grid[x][y] = z

        for row in grid:
            print(" | ".join([f"{str(val).rjust(3)}" for val in row]))
        print("-" * 30)

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, mooreMaps=None, diagonalMaps=None, hilbertMaps=None, ZMaps=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mooreMaps = mooreMaps
        self.diagonalMaps = diagonalMaps
        self.hilbertMaps = hilbertMaps
        self.ZMaps = ZMaps
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    def ZFlat(self, tensor, ZMap):
        B, C, h, w, d = tensor.shape
        device = tensor.device
        coord_map = ZMap.to(device)

        total_elements = h * w * d
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        i_in_frame = i_all % frame_size
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')

        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        flat = tensor[:, :, x_idx, y_idx, f_idx]
        return flat

    def ZReshape(self, flatTensor, ZMap):
        B, C, total_elements = flatTensor.shape
        device = flatTensor.device
        hw = ZMap.size(0)
        h = int(math.sqrt(hw))
        w = h
        d = total_elements // hw

        reshapeTensor = torch.zeros((B, C, h, w, d), dtype=flatTensor.dtype, device=device)
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')
        i_in_frame = i_all % frame_size

        coord_map = ZMap.to(device)
        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        reshapeTensor[:, :, x_idx, y_idx, f_idx] = flatTensor
        return reshapeTensor

    def hilbertFlat(self, tensor, hilbertMap):
        B, C, h, w, d = tensor.shape
        device = tensor.device
        coord_map = hilbertMap.to(device)

        total_elements = h * w * d
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        i_in_frame = i_all % frame_size
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')

        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        flat = tensor[:, :, x_idx, y_idx, f_idx]
        return flat

    def hilbertReshape(self, flatTensor, hilbertMap):
        B, C, total_elements = flatTensor.shape
        device = flatTensor.device
        hw = hilbertMap.size(0)
        h = int(math.sqrt(hw))
        w = h
        d = total_elements // hw

        reshapeTensor = torch.zeros((B, C, h, w, d), dtype=flatTensor.dtype, device=device)
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')
        i_in_frame = i_all % frame_size

        coord_map = hilbertMap.to(device)
        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        reshapeTensor[:, :, x_idx, y_idx, f_idx] = flatTensor
        return reshapeTensor


    def DiagonalFlat(self, tensor, diagonalMap):
        B, C, h, w, d = tensor.shape
        device = tensor.device
        coord_map = diagonalMap.to(device)

        total_elements = h * w * d
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        i_in_frame = i_all % frame_size
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')

        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        flat = tensor[:, :, x_idx, y_idx, f_idx]
        return flat

    def DiagonalReshape(self, flatTensor, diagonalMap):
        B, C, total_elements = flatTensor.shape
        device = flatTensor.device
        hw = diagonalMap.size(0)
        h = int(math.sqrt(hw))
        w = h
        d = total_elements // hw

        reshapeTensor = torch.zeros((B, C, h, w, d), dtype=flatTensor.dtype, device=device)
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')
        i_in_frame = i_all % frame_size

        coord_map = diagonalMap.to(device)
        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        reshapeTensor[:, :, x_idx, y_idx, f_idx] = flatTensor
        return reshapeTensor

    def mooreFlat(self, tensor, mooreMap):
        B, C, h, w, d = tensor.shape
        device = tensor.device
        coord_map = mooreMap.to(device)

        total_elements = h * w * d
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        i_in_frame = i_all % frame_size
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')

        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        flat = tensor[:, :, x_idx, y_idx, f_idx]
        return flat

    def mooreReshape(self, flatTensor, mooreMap):
        B, C, total_elements = flatTensor.shape
        device = flatTensor.device
        hw = mooreMap.size(0)
        h = int(math.sqrt(hw))
        w = h
        d = total_elements // hw

        reshapeTensor = torch.zeros((B, C, h, w, d), dtype=flatTensor.dtype, device=device)
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')
        i_in_frame = i_all % frame_size

        coord_map = mooreMap.to(device)
        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        reshapeTensor[:, :, x_idx, y_idx, f_idx] = flatTensor
        return reshapeTensor

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        img_dims_dhw = x.permute(0, 1, 4, 3, 2).shape[2:]

        # 后三维展平为一维进行扫描
        # 横向
        # x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # 竖向
        # x_flat_vertical = x.permute(0, 1, 4, 3, 2).reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat_vertical)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims_dhw).permute(0, 1, 3, 4, 2)

        # 斜向
        # frameSize = x.shape[-2]
        # diagonalMap = self.diagonalMaps[frameSize]
        # x_dgflat = self.DiagonalFlat(x, diagonalMap).transpose(-1, -2)
        # x_norm = self.norm(x_dgflat)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = self.DiagonalReshape(x_mamba.transpose(-1, -2), diagonalMap)

        # MRscan
        frameSize = x.shape[-2]
        mooreMap = self.mooreMaps[frameSize]
        x_MRflat = self.mooreFlat(x, mooreMap).transpose(-1, -2)
        x_norm = self.norm(x_MRflat)
        x_mamba = self.mamba(x_norm)
        # 恢复为原来的形状
        out = self.mooreReshape(x_mamba.transpose(-1, -2), mooreMap)

        # Hibertscan
        # frameSize = x.shape[-2]
        # hilbertMap = self.hilbertMaps[frameSize]
        # x_HBflat = self.mooreFlat(x, hilbertMap).transpose(-1, -2)
        # x_norm = self.norm(x_HBflat)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = self.hilbertReshape(x_mamba.transpose(-1, -2), hilbertMap)

        # z scan
        # frameSize = x.shape[-2]
        # ZMap = self.ZMaps[frameSize]
        # x_Zflat = self.ZFlat(x, ZMap).transpose(-1, -2)
        # x_norm = self.norm(x_Zflat)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = self.ZReshape(x_mamba.transpose(-1, -2), ZMap)

        out = out + x_skip

        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], mooreMaps=None,
                 diagonalMaps=None, hilbertMaps=None, ZMaps=None):
        super().__init__()

        self.mooreMaps = mooreMaps
        self.diagonalMaps = diagonalMaps
        self.hilbertMaps = hilbertMaps
        self.ZMaps = ZMaps
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i], mooreMaps=mooreMaps, diagonalMaps=diagonalMaps, hilbertMaps=hilbertMaps, ZMaps=ZMaps)
                  for j in
                  range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x



class emmamba(nn.Module):
    def __init__(
            self,
            in_chans=3,
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        mooreMaps = MooreCurve().mooreCurveMaps
        diagonalMaps = DiagonalMap().diagonalMaps
        hilbertMaps = Hilbert().hilbert_maps
        ZMaps = ZCurveMapper().maps

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                mooreMaps=mooreMaps,
                                diagonalMaps=diagonalMaps,
                                hilbertMaps=hilbertMaps,
                                ZMaps=ZMaps
                                )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=16, out_channels=self.out_chans)


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)

        x2 = outs[0]
        enc2 = self.encoder2(x2)

        x3 = outs[1]
        enc3 = self.encoder3(x3)

        x4 = outs[2]
        enc4 = self.encoder4(x4)

        enc_hidden = self.encoder5(outs[3])

        dec3 = self.decoder5(enc_hidden, enc4)

        dec2 = self.decoder4(dec3, enc3)

        dec1 = self.decoder3(dec2, enc2)

        dec0 = self.decoder2(dec1, enc1)

        out = self.decoder1(dec0)

        return self.out(out)



def clean_state_dict(state_dict):
    """
    Removes standard PyTorch Lightning prefixes from the state_dict keys
    to match the model's structure.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # Common Lightning prefixes to remove
        if name.startswith("model."):
            name = name[6:]
        elif name.startswith("net."):
            name = name[4:]
        elif name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict







