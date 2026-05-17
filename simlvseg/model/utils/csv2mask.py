import pandas as pd
import numpy as np
import skimage.draw
import cv2
import os

def create_mask_from_csv_dynamic(csv_path, video_filename, frame_number, height, width):
    """
    从CSV文件中读取多边形坐标，为指定的视频帧生成一个二进制掩码。

    这个函数模拟了您提供的 __getitem__ 函数中关于掩码生成的部分。

    参数:
    - csv_path (str): 标注信息所在的 CSV 文件的路径。
    - video_filename (str): 要为其生成掩码的视频文件名 (例如 '0X100009310A3BD7FC.avi')。
    - frame_number (int): 要生成掩码的特定帧的编号 (例如 46 或 61)。
    - height (int): 生成掩码的高度 (例如 112)。
    - width (int): 生成掩码的宽度 (例如 112)。

    返回:
    - np.ndarray: 一个形状为 (height, width) 的 NumPy 数组，
                  其中多边形区域内的值为 1，其他区域为 0。
                  如果找不到对应数据，则返回一个全零的掩码。
    """
    try:
        # 1. 使用 pandas 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 2. 根据视频文件名和帧编号筛选出对应的行
        trace_data = df[(df['FileName'] == video_filename) & (df['Frame'] == frame_number)]

        # 如果找不到任何数据，打印警告并返回一个空掩码
        if trace_data.empty:
            print(f"警告: 在 '{csv_path}' 中找不到文件 '{video_filename}' 和帧 {frame_number} 的数据。")
            return np.zeros((height, width), dtype=np.float32)

        # 3. 提取坐标列 (X1, Y1, X2, Y2) 并转换为 NumPy 数组
        #    这对应于 __getitem__ 中的 t = self.trace[...]
        coords = trace_data[['X1', 'Y1', 'X2', 'Y2']].to_numpy()

        # 4. 从坐标数据中构建闭合多边形的 x 和 y 顶点
        #    这部分逻辑与 __getitem__ 函数中的完全相同
        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        # 5. 创建一个全零的基础掩码
        mask = np.zeros((height, width), dtype=np.float32)

        # 6. 使用 scikit-image 在掩码上绘制实心多边形
        #    获取多边形内部所有像素的行(rr)和列(cc)索引
        rr, cc = skimage.draw.polygon(
            np.rint(y).astype(np.int64),  # 多边形的 y 坐标 (行)
            np.rint(x).astype(np.int64),  # 多边形的 x 坐标 (列)
            (height, width)               # 图像的尺寸
        )
        # 将多边形区域的像素值设为 1
        mask[rr, cc] = 1

        return mask

    except FileNotFoundError:
        print(f"错误: 无法找到文件 '{csv_path}'")
        return None
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


def create_mask_from_csv_pediatric(csv_path, video_filename, frame_number, height, width):
    """
    从CSV文件中读取多边形坐标，为指定的视频帧生成一个二进制掩码。

    这个函数模拟了您提供的 __getitem__ 函数中关于掩码生成的部分。

    参数:
    - csv_path (str): 标注信息所在的 CSV 文件的路径。
    - video_filename (str): 要为其生成掩码的视频文件名 (例如 '0X100009310A3BD7FC.avi')。
    - frame_number (int): 要生成掩码的特定帧的编号 (例如 46 或 61)。
    - height (int): 生成掩码的高度 (例如 112)。
    - width (int): 生成掩码的宽度 (例如 112)。

    返回:
    - np.ndarray: 一个形状为 (height, width) 的 NumPy 数组，
                  其中多边形区域内的值为 1，其他区域为 0。
                  如果找不到对应数据，则返回一个全零的掩码。
    """
    try:
        # 1. 使用 pandas 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 2. 根据视频文件名和帧编号筛选出对应的行
        trace_data = df[(df['FileName'] == video_filename) & (df['Frame'] == frame_number)]

        print(df['FileName'] == video_filename)

        # 如果找不到任何数据，打印警告并返回一个空掩码
        if trace_data.empty:
            print(f"警告: 在 '{csv_path}' 中找不到文件 '{video_filename}' 和帧 {frame_number} 的数据。")
            return np.zeros((height, width), dtype=np.float32)

        # 3. 提取坐标列 (X1, Y1, X2, Y2) 并转换为 NumPy 数组
        #    这对应于 __getitem__ 中的 t = self.trace[...]
        coords = trace_data[['X', 'Y']].to_numpy()

        # 4. 从坐标数据中构建闭合多边形的 x 和 y 顶点
        #    这部分逻辑与 __getitem__ 函数中的完全相同
        x, y = coords[:, 0], coords[:, 1]

        # 5. 创建一个全零的基础掩码
        mask = np.zeros((height, width), dtype=np.float32)

        # 6. 使用 scikit-image 在掩码上绘制实心多边形
        #    获取多边形内部所有像素的行(rr)和列(cc)索引
        rr, cc = skimage.draw.polygon(
            np.rint(y).astype(np.int64),  # 多边形的 y 坐标 (行)
            np.rint(x).astype(np.int64),  # 多边形的 x 坐标 (列)
            (height, width)              # 图像的尺寸
        )
        # 将多边形区域的像素值设为 1
        mask[rr, cc] = 1

        return mask

    except FileNotFoundError:
        print(f"错误: 无法找到文件 '{csv_path}'")
        return None
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


def create_mask_from_CAMUS(video_path, threshold=10):
    """
    读取CAMUS数据集的视频（或类似的分割GT视频），将其调整为128x128，
    并转换为由0和1组成的二进制掩码。

    参数:
        video_path (str): 视频文件的路径
        threshold (int):用于区分黑色背景和灰色区域的阈值。默认为10。
                        (像素值 < 10 被视为背景0，>= 10 被视为前景1)

    返回:
        np.array: 形状为 (帧数, 128, 128) 的二进制numpy数组 (uint8类型)
    """

    # 检查文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    masks = []

    while True:
        ret, frame = cap.read()

        # 如果没有读取到帧（视频结束），则退出循环
        if not ret:
            break

        # 1. 转换为灰度图 (以防视频是RGB格式读取的)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 调整大小为 128x128
        # 注意：对于分割掩码(Mask)，必须使用 INTER_NEAREST (最近邻插值)
        # 以避免产生模糊的边界值（例如0和1之间出现0.5）
        resized_frame = cv2.resize(gray_frame, (128, 128), interpolation=cv2.INTER_NEAREST)

        # 3. 创建二进制掩码 (0 和 1)
        binary_mask = np.where((resized_frame > 80) & (resized_frame < 120), 1, 0).astype(np.uint8)

        masks.append(binary_mask)

    cap.release()

    # 将列表转换为 numpy 数组
    return np.array(masks)