import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F

def video_to_tensor(video_path):
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 2. 检查是否成功打开视频
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return None

    # 3. 创建一个列表存储所有帧
    frames = []
    transform = transforms.ToTensor()  # 将帧转换为张量并归一化到 [0,1]

    # 4. 逐帧读取视频
    while True:
        ret, frame = cap.read()  # ret 是布尔值，frame 是读取的帧
        if not ret:
            break

        # 5. 将 BGR (OpenCV 默认格式) 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 6. 转换为 PyTorch 张量并添加到帧列表中
        frame_tensor = transform(frame)  # 转换为 (C, H, W)
        frames.append(frame_tensor)

    # 7. 释放视频捕获对象
    cap.release()

    # 8. 将帧列表转换为 4D 张量，并在第 0 维添加 batch 维度
    video_tensor = torch.stack(frames, dim=-1)  # 变成 (C, H, W, T)

    # 9. 添加 batch 维度，形状变成 (1, C, H, W, T)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor  # 返回形状为 (N=1, C, H, W, T) 的张量

def video_crop_to_tensor(video_path, crop_size=(112, 112)):
    """
    将视频文件转换为 PyTorch 张量，并裁剪每一帧到指定大小。

    参数:
    - video_path (str): 视频文件的路径。
    - crop_size (tuple): 裁剪后的高度和宽度，例如 (120, 120)。

    返回:
    - torch.Tensor: 形状为 (1, C, H, W, T) 的五维张量，若出错则返回 None。
    """
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 2. 检查是否成功打开视频
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return None

    # 3. 创建一个列表存储所有帧
    frames = []
    transform = transforms.ToTensor()  # 将帧转换为张量并归一化到 [0,1]

    # 4. 获取视频的原始帧尺寸
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 5. 计算裁剪区域的起始坐标（中心裁剪）
    crop_width, crop_height = crop_size
    if original_width < crop_width or original_height < crop_height:
        print(f"Error: Crop size {crop_size} is larger than original video size ({original_height}, {original_width})")
        cap.release()
        return None

    x_start = (original_width - crop_width) // 2
    y_start = (original_height - crop_height) // 2
    x_end = x_start + crop_width
    y_end = y_start + crop_height

    # 6. 逐帧读取视频
    while True:
        ret, frame = cap.read()  # ret 是布尔值，frame 是读取的帧
        if not ret:
            break

        # 7. 将 BGR (OpenCV 默认格式) 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 8. 裁剪帧
        cropped_frame = frame[y_start:y_end, x_start:x_end]

        # 9. 转换为 PyTorch 张量并添加到帧列表中
        frame_tensor = transform(cropped_frame)  # 转换为 (C, H, W)
        frames.append(frame_tensor)

    # 10. 释放视频捕获对象
    cap.release()

    if not frames:
        print("No frames were read from the video.")
        return None

    # 11. 将帧列表转换为 4D 张量，并在第 0 维添加 batch 维度
    video_tensor = torch.stack(frames, dim=-1)  # 变成 (C, H, W, T)

    # 12. 添加 batch 维度，形状变成 (1, C, H, W, T)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor  # 返回形状为 (1, C, H, W, T) 的张量


def video_resize_to_tensor(video_path, target_size=(128, 128)):
    """
    将视频文件转换为 PyTorch 张量，并调整每一帧的分辨率到指定大小。

    参数:
    - video_path (str): 视频文件的路径。
    - target_size (tuple): 目标分辨率，格式为 (高度, 宽度)，例如 (120, 120)。

    返回:
    - torch.Tensor: 形状为 (1, C, H, W, T) 的五维张量，若出错则返回 None。
    """
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 2. 检查是否成功打开视频
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return None

    # 3. 创建一个列表存储所有帧
    frames = []
    transform = transforms.ToTensor()  # 将帧转换为张量并归一化到 [0,1]

    # 4. 获取视频的原始帧尺寸
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original video size: (Height: {original_height}, Width: {original_width})")

    target_height, target_width = target_size

    # 5. 逐帧读取视频
    frame_count = 0
    while True:
        ret, frame = cap.read()  # ret 是布尔值，frame 是读取的帧
        if not ret:
            break

        # 6. 将 BGR (OpenCV 默认格式) 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 7. 调整帧的分辨率
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # 8. 转换为 PyTorch 张量并添加到帧列表中
        frame_tensor = transform(resized_frame)  # 转换为 (C, H, W)
        frames.append(frame_tensor)
        frame_count += 1

    # 9. 释放视频捕获对象
    cap.release()

    if not frames:
        print("No frames were read from the video.")
        return None

    print(f"Total frames read and resized: {frame_count}")

    # 10. 将帧列表转换为 4D 张量，并在第 0 维添加 batch 维度
    video_tensor = torch.stack(frames, dim=-1)  # 变成 (C, H, W, T)

    # 11. 添加 batch 维度，形状变成 (1, C, H, W, T)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor  # 返回形状为 (1, C, H, W, T) 的张量

def video_pad_to_tensor(video_path, target_size=(128, 128)):
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 2. 检查是否成功打开视频
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return None

    # 3. 创建一个列表存储所有帧
    frames = []
    transform = transforms.ToTensor()  # 将帧转换为张量并归一化到 [0,1]

    # 4. 获取视频的原始帧尺寸
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original video size: (Height: {original_height}, Width: {original_width})")

    target_height, target_width = target_size

    # 5. 逐帧读取视频
    frame_count = 0
    while True:
        ret, frame = cap.read()  # ret 是布尔值，frame 是读取的帧
        if not ret:
            break

        # 6. 将 BGR (OpenCV 默认格式) 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 7. 转换为 PyTorch 张量并使用0填充到目标形状
        frame_tensor = transform(frame)  # shape: (3, H, W)

        pad_h = (0, 16)  # 112 -> 128
        pad_w = (0, 16)  # 112 -> 128

        pad_frame_tensor = F.pad(frame_tensor, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), "constant",0)  # shape: (3, 128, 128)
        # 8. 转换为 PyTorch 张量并添加到帧列表中
        frames.append(pad_frame_tensor)
        frame_count += 1

    # 9. 释放视频捕获对象
    cap.release()

    if not frames:
        print("No frames were read from the video.")
        return None

    print(f"Total frames read and resized: {frame_count}")

    # 10. 将帧列表转换为 4D 张量，并在第 0 维添加 batch 维度
    video_tensor = torch.stack(frames, dim=-1)  # 变成 (C, H, W, T)

    # 11. 添加 batch 维度，形状变成 (1, C, H, W, T)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor  # 返回形状为 (1, C, H, W, T) 的张量

# 示例调用
video_path = 'your_video.avi'
video_tensor = video_to_tensor(video_path)
if video_tensor is not None:
    print(f"视频张量的形状: {video_tensor.shape}")  # 输出张量的形状


