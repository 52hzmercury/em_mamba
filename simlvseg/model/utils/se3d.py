import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeAndExcitation3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation3D, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // reduction, kernel_size=1),
            activation,
            nn.Conv3d(in_channel // reduction, out_channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        weighting = F.adaptive_avg_pool3d(x, 1)
        weighting = self.fc(weighting)
        x = self.conv(x)
        y = x * weighting
        return y


class SE3dBlock(nn.Module):
    """
    A 3D convolutional residual block implemented in PyTorch, equivalent to MONAI's ResidualUnit.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1, reduction=8, subunits=1, act='prelu', norm='instance', dropout=0.0):
        """
        Initialize the 3D residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            strides (int or tuple): Convolution strides.
            kernel_size (int or tuple): Size of the convolutional kernel.
            subunits (int): Number of convolution layers within the block.
            act (str): Activation function ('prelu' or 'relu').
            norm (str): Normalization type ('instance' or 'batch').
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.subunits = subunits
        self.kernel_size = kernel_size
        self.reduction = reduction

        # Activation function
        self.activation = nn.PReLU(out_channels) if act == 'prelu' else nn.ReLU(inplace=True)

        # Normalization
        norm_layer = nn.InstanceNorm3d if norm == 'instance' else nn.BatchNorm3d

        # Convolution layers within the block
        layers = []
        for i in range(subunits):
            conv_in = in_channels if i == 0 else out_channels
            conv_stride = strides if i == 0 else 1  # Apply stride to the first convolution only
            # layers.append(nn.Conv3d(conv_in, out_channels, kernel_size=kernel_size, stride=conv_stride, padding=kernel_size // 2, bias=True))
            layers.append(SqueezeAndExcitation3D(conv_in, out_channels, kernel_size=kernel_size, stride=conv_stride, reduction=reduction))
            layers.append(norm_layer(out_channels))
            layers.append(self.activation)
            if dropout > 0.0:
                layers.append(nn.Dropout3d(dropout))

        self.conv_block = nn.Sequential(*layers)

        # Residual connection (1x1x1 Conv if channels or strides don't match)
        if in_channels != out_channels or strides != 1:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=strides, bias=True)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # Residual connection
        residual = self.residual(x)
        # Main convolutional path
        out = self.conv_block(x)
        # Add residual connection
        return out + residual


# 程序入口
if __name__ == "__main__":
    # 示例：创建一个3D张量输入（例如，1个样本，4个通道，32x32x32的空间尺寸）
    input_tensor = torch.randn(1, 128, 112, 112, 32)  # 模拟一个3D图像数据输入
    model = SE3dBlock(128, 64)  # 假设输入有4个通道
    output = model(input_tensor)  # 前向传播

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
