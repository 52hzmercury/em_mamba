import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
# from monai.utils import alias, deprecated_arg, export

# from simlvseg.model.utils.kan import KANBlock

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128])
        self.encoder5 = self._create_encoder_block(128, [256, 256, 256])

        self.decoder4 = self._create_decoder_block(128 * 2, 128, False)
        self.decoder3 = self._create_decoder_block(64 * 2, 64, False)
        self.decoder2 = self._create_decoder_block(32 * 2, 32, False)
        self.decoder1 = self._create_decoder_block(16 * 2, 1, True)

        self.upconv5 = self._create_up_conv(256, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)

        self.maxpool = nn.MaxPool3d(2, 2)

    def _create_encoder_block(
            self,
            in_channel,
            channels,
            down_sampling=False,
    ):

        def _create_residual_unit(
                in_channels, out_channels, strides,
        ):
            return ResidualUnit(
                3,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                subunits=2,
                act=Act.PRELU,
                norm=Norm.INSTANCE,
                dropout=0.0,
                bias=True,
                adn_ordering="NDA",
            )

        _channels = [in_channel, *channels]

        units = []
        for i in range(len(channels) - 1):
            units.append(_create_residual_unit(_channels[i], _channels[i + 1], 1))
        units.append(
            _create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1)
        )

        return nn.Sequential(*units)

    def _create_decoder_block(
            self,
            in_channels,
            out_channels,
            is_top,
    ):
        res_unit = ResidualUnit(
            3,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=is_top,
            adn_ordering="NDA",
        )

        return res_unit

    def _create_up_conv(
            self,
            in_channels,
            out_channels,
    ):
        return Convolution(
            3,
            in_channels,
            out_channels,
            strides=2,
            kernel_size=3,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            is_transposed=True,
            adn_ordering="NDA",
        )

    def forward(self, x):
        _, _, h, w, d = x.shape

        if (h % 16 != 0) or (w % 16 != 0) or (d % 16 != 0):
            raise ValueError(f"Invalid volume size ({h}, {w}, {d}). The dimension need to be divisible by 16.")

        x1 = self.encoder1(x)
        x = self.maxpool(x1)

        x2 = self.encoder2(x)
        x = self.maxpool(x2)

        x3 = self.encoder3(x)
        x = self.maxpool(x3)

        x4 = self.encoder4(x)
        x = self.maxpool(x4)

        x = self.encoder5(x)  # [1, 128, 14, 14, 4] -> [1, 256, 7 ,7, 2]

        x = self.upconv5(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder4(x)

        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)


        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        # dx1 = x[:, :, :, :, :]

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)

        # 获取可视化图像
        # dx1 =  x
        # print(x.mean())

        # 导入可视化库
        # from simlvseg.model.utils.visualizer import ImageVisualizer
        # visualizer = ImageVisualizer()

        # 可视化多层图像
        # visualizer.show_images([x1[0, 0, :, :, 0], x2[0, 0, :, :, 0], x3[0, 0, :, :, 0], x4[0, 0, :, :, 0], x5[0, 0, :, :, 0],
        #                                x5[0, 100, :, :, 0], dx4[0, 0, :, :, 0], dx3[0, 0, :, :, 0], dx2[0, 0, :, :, 3], dx1[0, :, :, :, 3], ],
        #                         titles=['encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5', 'x5', 'dcoder4', 'dcoder3', 'decoder2', 'decoder1'])

        # visualizer.show_images([x5[0, 100, :, :, 0], dx4[0, 0, :, :, 0], dx3[0, 0, :, :, 0], dx2[0, 0, :, :, 3], dx1[0, :, :, :, 3]],
        #                        titles=['x5', 'dx4', 'dx3', 'dx2', 'dx1'])
        # 可视化单层图像
        # vis_img = dx1
        # image_list = []
        # for c in range(vis_img.shape[1]):
        #     c_img = vis_img[:, c, :, :, 0]
        #     image_list.append(c_img)
        # visualizer.show_images(image_list, cmap='jet', save_path='decoder3.png')
        # 可视化单张图像
        # visualizer.show_image(dx1[0, :, :, :, 0], cmap='jet', save_path='decoder1.png', colorbar=True)

        return x
        # return  -x


class UNet3DSmall(UNet3D):
    def __init__(self):
        super().__init__()

        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128])
        self.encoder5 = self._create_encoder_block(128, [256, 256])

        self.decoder4 = self._create_decoder_block(128 * 2, 128, False)
        self.decoder3 = self._create_decoder_block(64 * 2, 64, False)
        self.decoder2 = self._create_decoder_block(32 * 2, 32, False)
        self.decoder1 = self._create_decoder_block(16 * 2, 1, True)

        self.upconv5 = self._create_up_conv(256, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)

        self.maxpool = nn.MaxPool3d(2, 2)


# if __name__ == "__main__":
#     # 初始化模型
#     model = UNet3D().cuda(0)   # 或者使用 UNet3DSmall()
#     # model = OnlyUKAN3D().cuda(1)  # 或者使用 UNet3DSmall()
#     # 伪造输入数据：假设输入形状为 (batch_size=1, channels=3, depth=128, height=128, width=128)
#     # input = torch.randn(1, 3, 112, 112, 32).cuda(1)
#     input = torch.randn(1, 3, 112, 112, 16).cuda(0)
#
#     output = model(input)
#
#     from thop import profile
#
#     flops, params = profile(model, inputs=(input,))
#     print('Flops: ', flops, ', Params: ', params)
#     print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))
#
#     print(f"输出形状: {output.shape}")
#
#     # 测试真实数据
#     ####################################################################################################################
#
#     # step1 加载模型权重
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/camus/lightning_logs/version_1/checkpoints/epoch=48-step=1028.ckpt'
#
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/echonet-dynamic/lightning_logs/version_1/checkpoints/epoch=8-step=7922.ckpt'
#
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/pediatric/lightning_logs/version_2/checkpoints/epoch=6-step=2015.ckpt'
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool_ped/lightning_logs/version_1/checkpoints/last.ckpt'
#     checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool_ped/lightning_logs/version_0/checkpoints/last.ckpt'
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool_ped/lightning_logs/version_1/checkpoints/epoch=8-step=3357.ckpt'
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool_ped/lightning_logs/version_0/checkpoints/epoch=1-step=592.ckpt'
#
#
#
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool/lightning_logs/version_2/checkpoints/epoch=1-step=50.ckpt'
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool/lightning_logs/version_0/checkpoints/epoch=3-step=700.ckpt'
#     # checkpoint_path = r'/workdir3t/A-Echo/echo-barlowtwins/logs_dir/supervised_train/vis_pool/lightning_logs/version_3/checkpoints/epoch=19-step=500.ckpt'
#     checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=True)
#
#     # 获取 state_dict
#     state_dict = checkpoint['state_dict']
#
#     # 移除 'model.' 前缀
#     new_state_dict = {}
#     for key in state_dict:
#         new_key = key.replace("model.", "")  # 移除 'model.' 前缀
#         new_state_dict[new_key] = state_dict[key]
#
#     # step2 初始化模型并加载权重
#     model = UNet3D()
#     model.load_state_dict(new_state_dict)
#     model.eval()
#
#     # 将模型加载到 GPU 1
#     model = model.cuda(0)
#
#
#     # step3 加载真实数据
#     import numpy as np
#     from simlvseg.seg_3d.dynamic_dataset import Seg3DDataset
#     from simlvseg.model.utils.img2tensor import video_to_tensor, video_resize_to_tensor
#     from simlvseg.model.utils.result_vis import visualize_segmentation
#     from simlvseg.model.utils.csv2mask import create_mask_from_csv_dynamic, create_mask_from_csv_pediatric
#
#     # video_name = r'0X1D00424C7EBCCA52.avi'
#     video_name = r'0X1E19B51380EDA781.avi'
#     video_path = r'/workdir3t/A-Echo/echo-barlowtwins/data/EchoNet-Dynamic/Videos/' + video_name
#     # video_path = r'/workdir3t/A-Echo/echo-barlowtwins/data/pediatric_echo1/pediatric_echo/A4C/Videos/' + video_name
#     save_name = "dynamic_hard.avi"
#     # save_name = "dynamic_easy.avi"
#     ed_frame = 94
#     es_frame = -1
#
#
#     # get Echonet-Dynamic pt
#     ground_truth = create_mask_from_csv_dynamic(
#         r'/workdir3t/A-Echo/echo-barlowtwins/data/EchoNet-Dynamic/VolumeTracings.csv',
#         video_name,
#         ed_frame,
#         112,
#         112)
#
#     # get Echonet-Pediatric pt
#     # ground_truth = create_mask_from_csv_pediatric(
#     #     r'/workdir3t/A-Echo/echo-barlowtwins/data/pediatric_echo1/pediatric_echo/A4C/VolumeTracings.csv',
#     #     video_name,
#     #     ed_frame,
#     #     112,
#     #     112)
#
#     ground_truth = np.tile(ground_truth, (32, 1, 1))
#
#     # get CAMUS pt
#     # ground_truth_path = r'/workdir3t/A-Echo/hcc-echo-mae/data/CAMUS_public/a4c_112/patient0028_a4c_gt.npy'
#     # ground_truth = np.load(ground_truth_path)
#
#     video_tensor = video_resize_to_tensor(video_path)
#     video_tensor = video_tensor[..., ed_frame - 1:]
#
#     # 获取视频的帧数
#     num_frames = video_tensor.shape[-1]
#
#     # 如果视频帧数小于 32，使用镜像复制的方式填充
#     if num_frames < 32:
#         # 使用最后一帧进行镜像复制填充
#         repeat_frames = 32 - num_frames
#         video_tensor = torch.cat([video_tensor, video_tensor[:, :, :, :, -1:].repeat(1, 1, 1, 1, repeat_frames)],
#                                  dim=-1)
#
#     # 确保 input 取到前 16 帧
#     input = video_tensor[:, :, :, :, 0:32].cuda(0)
#
#     # step4 前向传播
#     output = model(input)
#
#     # 可视化第一帧
#     # gradcam(model, input, target_class=0)  # 选择类别，通常是背景或者目标类
#
#
#     # 统计参数量
#     # from thop import profile
#     #
#     # flops, params = profile(model, inputs=(input,))
#     # print('Flops: ', flops, ', Params: ', params)
#     # print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))
#
#     # 打印输出形状
#     print(f"输出形状: {output.shape}")
#
#     # 可视化并保存
#     visualize_segmentation(video_path, output, ground_truth, save_name, 1., 1., [0, 255, 0], [0, 0, 255])
#     # visualize_segmentation_jet(video_path, output, grount_truth, "375.avi")
#
#     # 保存输出的mask视频
#     # tensor_to_video_grayscale(output, 'output.avi')


def main():
    import os
    import time
    try:
        from thop import profile
        from thop import clever_format
    except ImportError:
        print("Error: 'thop' is not installed. Please run: pip install thop")
        return

    # 1. Configuration
    # NOTE: Update these parameters to match exactly what you used during training
    INPUT_SIZE = (1, 3, 128, 128, 16)  # (Batch, Channels, H, W, D)
    # CKPT_PATH = "/workdir1/cn24/program/SimLVSeg/lightning_logs/version_368/checkpoints/epoch=22-step=14260.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model
    print("Initializing model...")
    # Ensure these init parameters match your training config
    model = UNet3D().to(device)

    model.eval()

    # 4. Create Dummy Input
    # Mamba usually requires inputs on CUDA
    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # 5. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 30)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # 6. Calculate FLOPs
    print("-" * 30)
    print("Calculating FLOPs... (this might take a moment)")
    try:
        # thop.profile returns (flops, params)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, total_params], "%.3f")
        print(f"GFLOPs: {flops / 1e9:.3f} G")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")

    # 7. Calculate FPS
    print("-" * 30)
    print("Calculating FPS...")
    num_iterations = 50
    warmup = 10

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_iterations / total_time

    print(f"Average time per inference: {total_time / num_iterations:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print("-" * 30)


if __name__ == '__main__':
    main()