import cv2
import numpy as np
import torch
from PIL import Image


# def visualize_segmentation(video_path, segmentation_output, ground_truth_mask, output_video_path, alpha_seg=0.5, alpha_gt=0.5):
#     # 读取原始视频
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return
#
#     # 获取视频的一些信息
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # 创建视频写入对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (112, 112))
#
#     # 将segmentation_output转换为NumPy数组
#     segmentation_output = segmentation_output.squeeze(0)  # 删除第一个维度[1, 1, H, W, T] -> [1, H, W, T]
#     segmentation_output = segmentation_output.squeeze(0)  # 删除第二个维度[1, H, W, T] -> [H, W, T]
#
#     # 使用sigmoid转换logits为概率，并通过0.5阈值获得二值掩膜
#     segmentation_output = torch.sigmoid(segmentation_output)  # 归一化为[0, 1]
#     segmentation_output_bin = (segmentation_output > 0.5).float().cpu().numpy()  # 将其转为0或1的掩膜
#
#     # 检查 Ground Truth 掩膜有效性
#     if not np.all(np.isfinite(ground_truth_mask)):  # 检查是否有无效值
#         print("Ground Truth Mask contains invalid values.")
#         print(ground_truth_mask)
#         return
#
#     # 读取和处理每一帧
#     for t in range(16):
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 调整原始视频帧大小为112x112
#         frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)  # 将帧调整为目标分辨率（112x112）
#
#         # 获取当前帧的GT掩膜和网络分割结果
#         gt_mask = ground_truth_mask[t]  # [H, W] 格式的GT掩膜
#         seg_mask = segmentation_output_bin[:, :, t]  # 分割结果为二值化的掩膜
#
#         # 将GT掩膜和分割掩膜转为二值图像
#         gt_mask_bin = np.uint8(gt_mask * 255)
#         seg_mask_bin = np.uint8(seg_mask * 255)
#
#         # 创建一个与帧相同大小的空白图像
#         frame_with_seg = np.copy(frame)
#
#         # 将分割掩膜区域填充为红色（分割掩膜）
#         seg_color = np.zeros_like(frame)
#         seg_color[seg_mask_bin == 255] = [0, 0, 255]  # 红色：BGR(0, 0, 255)
#         frame_with_seg = cv2.addWeighted(frame_with_seg, 1.0, seg_color, alpha_seg, 0)
#
#         # 将GT掩膜区域填充为绿色（GT掩膜）
#         gt_color = np.zeros_like(frame)
#         gt_color[gt_mask_bin == 255] = [0, 255, 0]  # 绿色：BGR(0, 255, 0)
#         frame_with_seg = cv2.addWeighted(frame_with_seg, 1.0, gt_color, alpha_gt, 0)
#
#         # 写入结果到输出视频
#         out.write(frame_with_seg)
#
#     # 释放资源
#     cap.release()
#     out.release()
#     print("处理完成，结果视频已保存为", output_video_path)


# def visualize_segmentation_jet(video_path, segmentation_output, ground_truth_mask, output_video_path, alpha_seg=0.7, alpha_gt=0.5):
#     # 读取原始视频
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return
#
#     # 获取视频的一些信息
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # 创建视频写入对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (112, 112))
#
#     # 将segmentation_output转换为NumPy数组
#     segmentation_output = segmentation_output.squeeze(0)  # 删除第一个维度[1, 1, H, W, T] -> [1, H, W, T]
#     segmentation_output = segmentation_output.squeeze(0)  # 删除第二个维度[1, H, W, T] -> [H, W, T]
#
#     # 使用sigmoid转换logits为概率，并通过0.5阈值获得二值掩膜
#     segmentation_output = torch.sigmoid(segmentation_output).detach().float().cpu().numpy()
#
#     # 检查 Ground Truth 掩膜有效性
#     if not np.all(np.isfinite(ground_truth_mask)):  # 检查是否有无效值
#         print("Ground Truth Mask contains invalid values.")
#         print(ground_truth_mask)
#         return
#
#     # 读取和处理每一帧
#     for t in range(16):
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 调整原始视频帧大小为112x112
#         frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)  # 将帧调整为目标分辨率（112x112）
#
#         # 获取当前帧的GT掩膜和网络分割结果
#         seg_mask = segmentation_output[:, :, t]  # 分割结果为二值化的掩膜
#
#         # 确保 seg_mask 是 uint8
#         seg_mask = (seg_mask * 255).astype(np.uint8)
#
#         # 将GT掩膜和分割掩膜转为二值图像
#         seg_mask_bin = cv2.applyColorMap(seg_mask, cv2.COLORMAP_JET)
#
#         # 创建一个与帧相同大小的空白图像
#         frame_with_seg = np.copy(frame)
#
#         # 将分割掩膜区域填充为红色（分割掩膜）
#         frame_with_seg = cv2.addWeighted(frame_with_seg, 1.0, seg_mask_bin, alpha_seg, 0)
#
#         # 写入结果到输出视频
#         out.write(frame_with_seg)
#
#     # 释放资源
#     cap.release()
#     out.release()
#     print("处理完成，结果视频已保存为", output_video_path)


import cv2
import numpy as np
import torch


def calculate_iou(seg_mask_bin, gt_mask_bin):
    # 计算交集（Intersection）和并集（Union）
    intersection = np.sum(np.logical_and(seg_mask_bin, gt_mask_bin))
    union = np.sum(np.logical_or(seg_mask_bin, gt_mask_bin))

    # 防止除零错误
    if union == 0:
        return 0.0
    else:
        return intersection / union


# def visualize_segmentation(video_path, segmentation_output, ground_truth_mask, output_video_path, alpha_seg=0.5, alpha_gt=0.5):
#     # 读取原始视频
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return
#
#     # 获取视频的一些信息
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # 创建视频写入对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (112, 112))
#
#     # 将segmentation_output转换为NumPy数组
#     segmentation_output = segmentation_output.squeeze(0)  # 删除第一个维度[1, 1, H, W, T] -> [1, H, W, T]
#     segmentation_output = segmentation_output.squeeze(0)  # 删除第二个维度[1, H, W, T] -> [H, W, T]
#
#     # 使用sigmoid转换logits为概率，并通过0.5阈值获得二值掩膜
#     segmentation_output = torch.sigmoid(segmentation_output)  # 归一化为[0, 1]
#     segmentation_output_bin = (segmentation_output > 0.5).float().cpu().numpy()  # 将其转为0或1的掩膜
#
#     # 检查 Ground Truth 掩膜有效性
#     if not np.all(np.isfinite(ground_truth_mask)):  # 检查是否有无效值
#         print("Ground Truth Mask contains invalid values.")
#         print(ground_truth_mask)
#         return
#
#     # 处理ground_truth_mask的帧数，使其符合32帧要求
#     gt_frames = ground_truth_mask.shape[0]
#     if gt_frames != 32:
#         print(f"Ground truth mask has {gt_frames} frames. Mirroring to 32 frames.")
#
#         if gt_frames < 32:
#             # 使用镜像复制来填充不足的帧数
#             pad_size = 32 - gt_frames
#             mirrored_part = np.flip(ground_truth_mask[:pad_size], axis=0)  # 镜像复制
#             ground_truth_mask = np.concatenate([ground_truth_mask, mirrored_part], axis=0)
#         elif gt_frames > 32:
#             # 截断到32帧
#             ground_truth_mask = ground_truth_mask[:32]
#
#     # 读取和处理每一帧
#     for t in range(segmentation_output.shape[-1]):
#         ret, frame = cap.read()
#
#         # 调整原始视频帧大小为112x112
#         frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)  # 将帧调整为目标分辨率（112x112）
#
#         # 获取当前帧的GT掩膜和网络分割结果
#         gt_mask = ground_truth_mask[t]  # [H, W] 格式的GT掩膜
#         seg_mask = segmentation_output_bin[:, :, t]  # 分割结果为二值化的掩膜
#
#         # 将GT掩膜和分割掩膜转为二值图像
#         gt_mask_bin = np.uint8(gt_mask * 255)
#         seg_mask_bin = np.uint8(seg_mask * 255)
#
#         # 创建一个与帧相同大小的空白图像
#         frame_with_seg = np.copy(frame)
#
#         # 将分割掩膜区域填充为红色（分割掩膜）
#         seg_color = np.zeros_like(frame)
#         seg_color[seg_mask_bin == 255] = [0, 0, 255]  # 红色：BGR(0, 0, 255)
#         frame_with_seg = cv2.addWeighted(frame_with_seg, 1.0, seg_color, alpha_seg, 0)
#
#         # 将GT掩膜区域填充为绿色（GT掩膜）
#         gt_color = np.zeros_like(frame)
#         gt_color[gt_mask_bin == 255] = [0, 255, 0]  # 绿色：BGR(0, 255, 0)
#         frame_with_seg = cv2.addWeighted(frame_with_seg, 1.0, gt_color, alpha_gt, 0)
#
#         # 计算IOU
#         iou = calculate_iou(seg_mask_bin, gt_mask_bin)
#         print(f"{t + 1}: {iou:.4f}")
#
#         # 写入结果到输出视频
#         out.write(frame_with_seg)
#
#     # 释放资源
#     cap.release()
#     out.release()
#     print("处理完成，结果视频已保存为", output_video_path)

def visualize_segmentation(video_path,
                                        segmentation_output,
                                        ground_truth_mask,
                                        output_video_path,
                                        alpha_seg=0.5,
                                        alpha_gt=0.1,
                                        gt_color_bgr=[122, 200, 121],       # 绿色 (Ground Truth Only)
                                        seg_color_bgr=[134, 90, 190],       # 红色 (Segmentation Only)
                                        overlap_color_bgr=[255, 223, 128]   # 蓝色 (Overlap)
                                       ):
    """
    使用NumPy处理掩码叠加，并将重叠区域显示为黄色。
    """
    # 1. 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # 2. 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_size = (128, 128)

    # 3. 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    # 4. 准备分割掩码 (T, H, W)
    with torch.no_grad():
        squeezed_output = segmentation_output.squeeze()
        if squeezed_output.shape == (128, 128, 32):
             squeezed_output = squeezed_output.permute(2, 0, 1)

        probs = torch.sigmoid(squeezed_output)
        seg_mask_binary = (probs > 0.5).cpu().numpy().astype(bool)

    # 5. 准备Ground Truth掩码并确定帧数
    ground_truth_mask = ground_truth_mask.astype(bool)
    num_gt_frames = ground_truth_mask.shape[0]
    num_seg_frames = seg_mask_binary.shape[0]
    num_frames_to_process = min(num_gt_frames, num_seg_frames)

    # 6. 逐帧处理和叠加
    for t in range(num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 视频在第 {t+1} 帧提前结束。")
            break

        #camus
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frame_float = frame_resized.astype(np.float32)

        # echo
        # 获取原始帧的尺寸
        original_h, original_w = frame.shape[:2]
        # 创建128x128的画布并进行零填充
        # frame_padded = np.zeros((128, 128, 3), dtype=np.uint8)
        # frame_padded[:original_h, :original_w, :] = frame
        # frame_float = frame_padded.astype(np.float32)

        gt_mask = ground_truth_mask[t]
        seg_mask = seg_mask_binary[t]

        # --- 核心改动：计算三个互斥的区域 ---
        # 1. 计算重叠区域 (Intersection)
        intersection_mask = np.logical_and(gt_mask, seg_mask)

        # 2. 计算仅存在于GT的区域 (GT Only)
        gt_only_mask = np.logical_and(gt_mask, np.logical_not(seg_mask))

        # 3. 计算仅存在于分割结果的区域 (Segmentation Only)
        seg_only_mask = np.logical_and(seg_mask, np.logical_not(gt_mask))

        # --- 对三个区域分别进行颜色混合 ---
        # 使用NumPy的布尔索引高效地为不同区域上色

        # 为 "GT Only" 区域上绿色
        frame_float[gt_only_mask] = \
            frame_float[gt_only_mask] * (1 - alpha_gt) + np.array(gt_color_bgr, dtype=np.float32) * alpha_gt

        # 为 "Segmentation Only" 区域上红色
        frame_float[seg_only_mask] = \
            frame_float[seg_only_mask] * (1 - alpha_seg) + np.array(seg_color_bgr, dtype=np.float32) * alpha_seg

        # 为重叠区域上黄色，透明度可以取两者平均或最大值，这里取最大值
        alpha_overlap = max(alpha_gt, alpha_seg)
        frame_float[intersection_mask] = \
            frame_float[intersection_mask] * (1 - alpha_overlap) + np.array(overlap_color_bgr, dtype=np.float32) * alpha_overlap

        # 将计算后的浮点图像转换回uint8以便保存
        final_frame = np.clip(frame_float, 0, 255).astype(np.uint8)

        # 计算并绘制 IOU
        iou = calculate_iou(seg_mask, gt_mask)
        print(f"IOU: {iou:.4f}")
        # cv2.putText(final_frame, f"IOU: {iou:.4f}", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 写入视频帧
        out.write(final_frame)

    # 7. 释放资源
    cap.release()
    out.release()
    print(f"处理完成！重叠区域已高亮显示，结果保存至: {output_video_path}")
