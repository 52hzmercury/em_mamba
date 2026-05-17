import torch
import torch.nn as nn


# 论文地址：https://arxiv.org/pdf/2303.16900
# 论文：InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)
class InceptionDWConv3d(nn.Module):
    """ Inception depthwise convolution for 3D data
    """

    def __init__(self, in_channels, out_channels, stride=2, cube_kernel_size=3, spatial_kernel_size=7, temporal_kernel_size=7, branch_ratio=1/8):
        super().__init__()

        in_gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        out_gc = int(out_channels * (1/8))
        self.dwconv_hwd = nn.Conv3d(in_gc, out_gc, stride=stride, kernel_size=cube_kernel_size, padding=cube_kernel_size // 2, groups=in_gc)
        self.dwconv_wd = nn.Conv3d(in_gc, out_gc, stride=stride, kernel_size=(1, 1, temporal_kernel_size), padding=(0, 0, temporal_kernel_size // 2),
                                   groups=in_gc)
        self.dwconv_hd = nn.Conv3d(in_gc, out_gc, stride=stride, kernel_size=(1, spatial_kernel_size, 1), padding=(0, temporal_kernel_size // 2, 0),
                                   groups=in_gc)
        self.dwconv_hw = nn.Conv3d(in_gc, out_gc, stride=stride, kernel_size=(spatial_kernel_size, 1, 1), padding=(spatial_kernel_size // 2, 0, 0),
                                   groups=in_gc)
        # self.split_indexes = (in_channels - 4 * in_gc, in_gc, in_gc, in_gc, in_gc)
        self.split_indexes = (in_channels - 4 * in_gc, in_gc, in_gc, in_gc, in_gc)

        self.dwconv_res = nn.Conv3d(in_channels - 4 * in_gc, out_channels - 4 * out_gc, stride=stride, kernel_size=(1, 1, 1))

    def forward(self, x):
        x_id, x_hwd, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (self.dwconv_res(x_id), self.dwconv_hwd(x_hwd), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd), self.dwconv_hw(x_hw)),
            dim=1,
        )


class InceptionDWConv3dBlock(nn.Module):
    """
    A 3D convolutional residual block implemented in PyTorch, equivalent to MONAI's ResidualUnit.
    """

    def __init__(self, in_channels, out_channels, strides=1, cube_kernel_size=3, spatial_kernel_size=7, temporal_kernel_size=7, subunits=2, act='prelu', norm='instance', dropout=0.0):
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
            layers.append(InceptionDWConv3d(conv_in, out_channels, stride=conv_stride, cube_kernel_size=cube_kernel_size, spatial_kernel_size=spatial_kernel_size, temporal_kernel_size=temporal_kernel_size))
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


# Example usage
if __name__ == "__main__":
    # Define the 3D residual module with example parameters
    module = InceptionDWConv3dBlock(in_channels=64, out_channels=32, strides=2, subunits=1, act='prelu', norm='instance', dropout=0.0)

    # Example input tensor with shape (batch_size=1, channels=16, depth=16, height=128, width=128)
    input_tensor = torch.randn(1, 64, 16, 128, 128)

    # Forward pass
    output_tensor = module(input_tensor)

    # Print the input and output tensor shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")



# if __name__ == '__main__':
#     block = InceptionDWConv3d(64, 128)  # 输入 C
#     input = torch.randn(1, 64, 16, 224, 224)  # 输入B C D H W
#     output = block(input)
#     print(input.size())
#     print(output.size())
