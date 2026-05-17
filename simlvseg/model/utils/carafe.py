import torch
import torch.nn as nn
from torch.nn import functional as F


def pixel_shuffle_3d(input, upscale_factor):
    """
    自定义 3D pixel shuffle 操作。

    :param input: 输入张量，形状为 [N, C, H, W, D]
    :param upscale_factor: 上采样因子
    :return: 输出张量，形状为 [N, C // (upscale_factor^3), H*upscale_factor, W*upscale_factor, D*upscale_factor]
    """
    N, C, H, W, D = input.size()
    r = upscale_factor

    # 检查通道数是否可以被 r^3 整除
    assert C % (r ** 3) == 0, f"输入通道数 {C} 必须能被 upscale_factor^3 ({r ** 3}) 整除"

    # 新的通道数
    new_C = C // (r ** 3)

    # 调整形状: 将通道维度展开到空间维度
    output = input.view(N, new_C, r, r, r, H, W, D)

    # 交换维度: 将 (r, r, r) 转移到 (H, W, D)
    output = output.permute(0, 1, 5, 2, 6, 3, 7, 4)

    # 合并空间维度
    output = output.reshape(N, new_C, H * r, W * r, D * r)

    return output


class CARAFE(nn.Module):
    # CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor


class CARAFE3D(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE3D, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor

        # 降维模块，用1x1x1卷积减少通道数到 c1 // 4
        self.down = nn.Conv3d(c1, c1 // 4, kernel_size=1)

        # 编码生成上采样卷积核
        self.encoder = nn.Conv3d(
            c1 // 4,
            self.up_factor ** 3 * self.kernel_size ** 3,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

        # 输出模块，用1x1x1卷积调整输出通道数到 c2
        self.out = nn.Conv3d(c1, c2, kernel_size=1)

    def forward(self, x):
        """
        前向传播函数，输入为 [N, C, H, W, D]
        """
        N, C, H, W, D = x.size()

        # Kernel Prediction Module
        kernel_tensor = self.down(x)  # 降维操作
        kernel_tensor = self.encoder(kernel_tensor)  # 编码生成上采样卷积核
        kernel_tensor = kernel_tensor.contiguous()  # 确保内存连续性
        kernel_tensor = pixel_shuffle_3d(kernel_tensor, self.up_factor)  # 像素重排操作
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # 权重归一化

        # unfold 操作适配 3D 空间
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(4, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(
            N, self.kernel_size ** 3, H, W, D, self.up_factor ** 3
        )
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 4, 1, 5)

        # Content-Aware ReAssembly Module
        x = F.pad(
            x,
            pad=(self.kernel_size // 2,) * 6,
            mode='constant',
            value=0
        )
        x = x.unfold(2, self.kernel_size, step=1)
        x = x.unfold(3, self.kernel_size, step=1)
        x = x.unfold(4, self.kernel_size, step=1)
        x = x.reshape(N, C, H, W, D, -1)
        x = x.permute(0, 2, 3, 4, 1, 5)

        out_tensor = torch.matmul(x, kernel_tensor)  # 点积操作
        out_tensor = out_tensor.reshape(N, H, W, D, -1)
        out_tensor = out_tensor.permute(0, 4, 1, 2, 3)
        out_tensor = out_tensor.contiguous()  # 确保内存连续性

        # 调试打印 Pixel Shuffle 之前的张量形状
        print("Pixel Shuffle 前形状:", out_tensor.shape)

        out_tensor = pixel_shuffle_3d(out_tensor, self.up_factor)  # 像素重排操作

        # 调试打印 Pixel Shuffle 之后的张量形状
        print("Pixel Shuffle 后形状:", out_tensor.shape)

        out_tensor = self.out(out_tensor)  # 调整输出通道数

        return out_tensor  # 返回上采样后的特征图





# if __name__ == "__main__":
#     # Test the CARAFE module
#     input_tensor = torch.randn(1, 64, 32, 32)  # Example input tensor (batch size 1, 64 channels, 32x32 size)
#     model = CARAFE(c1=64, c2=128, kernel_size=3, up_factor=2)  # Instantiate CARAFE
#     output_tensor = model(input_tensor)  # Forward pass
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor.shape)


if __name__ == "__main__":
    # 测试 CARAFE3D 模块
    N, C, H, W, D = 1, 64, 32, 32, 64  # 输入特征图维度
    x = torch.randn(N, C, H, W, D)  # 随机输入特征图

    carafe_3d = CARAFE3D(c1=64, c2=32, kernel_size=3, up_factor=2)  # 创建CARAFE3D模块
    output = carafe_3d(x)  # 前向传播
    print("输出形状:", output.shape)  # 输出形状应为 [N, c2, S*H, S*W, S*D]
