import numpy as np
import os
import copy
import torch
import torch.nn.functional as F

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: T, H, W, C
    T, H, W, C = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[begin+bias:end-bias, :, :, :]# center_crop
        cropped_length = data.shape[0]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[begin+bias:begin+bias+cropped_length, :, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(3, 1, 2, 0).contiguous().view(C * H * W, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * H * W, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, H, W, window).permute(3, 1, 2, 0).contiguous().numpy()

    return data

def expand_and_mirror(X, M):
    N, H, W, C = X.shape

    assert M > N

    # Calculate the total number of repetitions needed to exceed or meet M
    repeat_factor = (M + N - 1) // N  # Ceiling division to ensure at least M elements

    # Generate a mirrored sequence of indices
    mirrored_indices = np.arange(N)
    for _ in range(repeat_factor // 2):
        mirrored_indices = np.concatenate([mirrored_indices, mirrored_indices[::-1]])

    # If the repeat factor is odd, add one more direct copy of the original indices
    if repeat_factor % 2 != 0:
        mirrored_indices = np.concatenate([mirrored_indices, np.arange(N)])

    # Ensure the mirrored sequence is at least of length M and truncate if necessary
    mirrored_indices = mirrored_indices[:M]

    # Use advanced indexing to create the expanded and mirrored array
    expanded_X = X[mirrored_indices]

    return expanded_X

def pad_array(X, M):
    """
    Pad the numpy array X of shape (N, H, W, 3) to a new shape (M, H, W, 3)
    by adding padding on both edges of the first dimension.

    Parameters:
    - X: Input array of shape (N, H, W, 3)
    - M: The new size of the first dimension after padding

    Returns:
    - Padded array of shape (M, H, W, 3)
    """
    N, H, W = X.shape[:3]  # Original dimensions of X

    # Calculate the total padding needed
    total_pad = M - N
    # Ensure that the total padding is non-negative
    if total_pad < 0:
        raise ValueError("M must be greater than N")

    # Calculate padding for the beginning and end of the first dimension
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before

    # Create padding configuration
    if len(X.shape) == 4:
        pad_width = [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)]
    elif len(X.shape) == 3:
        pad_width = [(pad_before, pad_after), (0, 0), (0, 0)]

    # Apply padding
    padded_X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)

    return padded_X

def pad_array_with_images(X, M):
    """
    Pad the numpy array X of shape (N, H, W, 3) to a new shape (M, H, W, 3)
    by adding the first and last image on both edges of the first dimension.

    Parameters:
    - X: Input array of shape (N, H, W, 3)
    - M: The new size of the first dimension after padding

    Returns:
    - Padded array of shape (M, H, W, 3)
    """
    N, H, W, C = X.shape  # Original dimensions of X

    # Calculate the total padding needed
    total_pad = M - N
    # Ensure that the total padding is non-negative
    if total_pad < 0:
        raise ValueError("M must be greater than N")

    # Calculate padding for the beginning and end of the first dimension
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before

    # Replicate the first and last image for padding
    pad_before_images = np.repeat(X[:1], pad_before, axis=0)
    pad_after_images = np.repeat(X[-1:], pad_after, axis=0)

    # Concatenate the padding and the original array
    padded_X = np.concatenate([pad_before_images, X, pad_after_images], axis=0)

    return padded_X


def pad_array_with_origin_images_seq(X, M):
    """
    将输入数组 a4c_seq 的第一维扩展到 128，通过重复后三维数据实现。

    参数:
    a4c_seq (numpy.ndarray): 形状为 (22, 112, 112, 3) 的数组。

    返回:
    numpy.ndarray: 形状为 (128, 112, 112, 3) 的数组。
    """
    # 计算需要重复的次数
    num_repeats = -(-M // X.shape[0])  # 向上取整

    # 重复数组
    repeated_array = np.tile(X, (num_repeats, 1, 1, 1))

    # 截取前 target_length 个元素
    padded_array = repeated_array[:M]

    return padded_array


def pad_array_with_origin_images_gt(X, M):
    """
    将输入数组 a4c_gt 的第一维扩展到指定的目标长度，通过重复后二维数据实现。

    参数:
    a4c_gt (numpy.ndarray): 形状为 (22, 112, 112) 的数组。
    M (int): 目标长度，即扩展后的数组第一维的长度。

    返回:
    numpy.ndarray: 形状为 (M, 112, 112) 的数组。
    """
    # 计算需要重复的次数
    num_repeats = -(-M // X.shape[0])  # 向上取整

    # 重复数组
    repeated_array = np.tile(X, (num_repeats, 1, 1))

    # 截取前 M 个元素
    padded_array = repeated_array[:M]

    return padded_array


class CAMUSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_frames, mean, std, patient_names):
    # def __init__(self, data_dir, n_frames, mean, std):
        self.data_dir = data_dir
        self.n_frames = n_frames

        # self.patients = sorted(
        #     [filename.split('_')[0] for filename in os.listdir(self.data_dir) if '_gt.npy' in filename])

        # 读取文件初始化
        self.patients = patient_names

        self.mean = np.array(mean)
        self.std = np.array(std)


    def __len__(self):
        return len(self.patients)

    def pad_to_shape(self, arr):
        """
        将输入数组 arr 的图像数据从右边和底边填充为 128。

        参数:
        arr (numpy.ndarray): 输入数组，形状为 (F, H, W, C ) 或 (F, H, W)。

        返回:
        numpy.ndarray: 填充后的数组，形状为 (F, 128, 128, C) 或 (F, 128, 128)。
        """
        if arr.ndim == 4:
            F, H, W, C= arr.shape
        elif arr.ndim == 3:
            F, H, W = arr.shape
            C = 1
        else:
            raise ValueError("输入数组的维度必须为 3 或 4")

        # 计算需要填充的大小
        pad_height = 128 - H
        pad_width = 128 - W

        if pad_height < 0 or pad_width < 0:
            raise ValueError("输入数组的尺寸不能大于 128x128")

        # 创建填充配置
        if arr.ndim == 4:
            pad_widths = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        elif arr.ndim == 3:
            pad_widths = ((0, 0), (0, pad_height), (0, pad_width))

        # 应用填充
        padded_arr = np.pad(arr, pad_width=pad_widths, mode='constant', constant_values=0)

        return padded_arr

    def __getitem__(self, idx):
        patient = self.patients[idx]

        a4c_seq = np.load(os.path.join(self.data_dir, f'{patient}_a4c_seq.npy'))
        a4c_gt = np.load(os.path.join(self.data_dir, f'{patient}_a4c_gt.npy'))

        a4c_seq = np.float32(a4c_seq) / 255.
        a4c_gt = np.float32(a4c_gt)

        if len(a4c_seq.shape) == 3:
            a4c_seq = a4c_seq[..., np.newaxis] * np.ones((1, 1, 1, 3))

        a4c_seq = (a4c_seq - self.mean) / self.std


        # 使用填充函数填充到128
        a4c_seq = self.pad_to_shape(a4c_seq)
        a4c_gt = self.pad_to_shape(a4c_gt)


        if self.n_frames != a4c_seq.shape[0]:
            # TYPE 1: MIRRORING
            # a4c_seq = expand_and_mirror(a4c_seq, self.n_frames)
            # tmp = np.zeros_like(a4c_seq)[...,0]
            # tmp[:a4c_gt.shape[0]] = a4c_gt
            # a4c_gt = tmp.copy()

            # TYPE 2: PADDING TO ZERO
            # valid_frame_num = np.sum(a4c_seq.sum(1).sum(-1).sum(-1) != 0)
            # # 112
            # # tmp = np.zeros((self.n_frames, 112, 112, 3)).astype(a4c_seq.dtype)
            # # 128
            # tmp = np.zeros((self.n_frames, 128, 128, 3)).astype(a4c_seq.dtype)
            # tmp[:a4c_seq.shape[0]] = a4c_seq
            # a4c_seq = tmp.copy()
            # tmp = np.zeros_like(a4c_seq)[..., 0]
            # tmp[:a4c_gt.shape[0]] = a4c_gt
            # a4c_gt = tmp.copy()

            # TYPE 3:
            # a4c_seq = pad_array(a4c_seq, self.n_frames)
            # a4c_gt  = pad_array(a4c_gt,  self.n_frames)

            # TYPE 4:
            # a4c_seq = pad_array_with_images(a4c_seq, self.n_frames)
            # a4c_gt  = pad_array(a4c_gt,  self.n_frames)

            # TYPE 5:
            # a4c_seq = pad_array_with_origin_images_seq(a4c_seq, self.n_frames)
            # a4c_gt = pad_array_with_origin_images_gt(a4c_gt, self.n_frames)

            # TYPE 6:
            valid_frame_num = np.sum(a4c_seq.sum(1).sum(-1).sum(-1) != 0)
            a4c_seq = valid_crop_resize(a4c_seq, valid_frame_num, [1], self.n_frames)
            a4c_gt = a4c_gt[:, :, :, None]
            a4c_gt = valid_crop_resize(a4c_gt, valid_frame_num,[1], self.n_frames)[...,0]  # [...，0]中..表示保留了前面的所有维度，0表示选择最后一维的第一个元荼

        else:
            valid_frame_num = self.n_frames

        assert a4c_seq.shape[0] == self.n_frames

        # (F, H, W, C) --> (C, H, W, F)
        a4c_seq = a4c_seq.transpose((3, 1, 2, 0))
        a4c_gt = a4c_gt.transpose((1, 2, 0))

        a4c_seq = np.float32(a4c_seq)
        a4c_gt = np.float32(a4c_gt)

        # 获取ED和ES的标注
        # a4c_gt = {
        #     'label_index': [0, int(valid_frame_num - 1)],
        #     'video_gt': a4c_gt,
        # }

        #all
        a4c_gt = {
            'label_index': [0, self.n_frames - 1],
            'video_gt': a4c_gt,
        }

        # 实际视频数据 标注数据 病人序号
        return a4c_seq, a4c_gt, patient




class CAMUSDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_frames, mean, std, patient_names):
        self.data_dir = data_dir
        self.n_frames = n_frames

        # self.patients = sorted(
        #     [filename.split('_')[0] for filename in os.listdir(self.data_dir) if '_gt.npy' in filename])

        # 读取文件初始化
        self.patients = patient_names

        self.mean = np.array(mean)
        self.std = np.array(std)

    def __len__(self):
        return len(self.patients)
    def pad_to_shape(self, arr):
        """
        将输入数组 arr 的图像数据从右边和底边填充为 128。

        参数:
        arr (numpy.ndarray): 输入数组，形状为 (F, H, W, C ) 或 (F, H, W)。

        返回:
        numpy.ndarray: 填充后的数组，形状为 (F, 128, 128, C) 或 (F, 128, 128)。
        """
        if arr.ndim == 4:
            F, H, W, C= arr.shape
        elif arr.ndim == 3:
            F, H, W = arr.shape
            C = 1
        else:
            raise ValueError("输入数组的维度必须为 3 或 4")

        # 计算需要填充的大小
        pad_height = 128 - H
        pad_width = 128 - W

        if pad_height < 0 or pad_width < 0:
            raise ValueError("输入数组的尺寸不能大于 128x128")

        # 创建填充配置
        if arr.ndim == 4:
            pad_widths = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        elif arr.ndim == 3:
            pad_widths = ((0, 0), (0, pad_height), (0, pad_width))

        # 应用填充
        padded_arr = np.pad(arr, pad_width=pad_widths, mode='constant', constant_values=0)

        return padded_arr
    def __getitem__(self, idx):
        patient = self.patients[idx]

        a4c_seq = np.load(os.path.join(self.data_dir, f'{patient}_a4c_seq.npy'))
        a4c_gt  = np.load(os.path.join(self.data_dir, f'{patient}_a4c_gt.npy'))

        a4c_seq = np.float32(a4c_seq) / 255.
        a4c_gt  = np.float32(a4c_gt)

        if len(a4c_seq.shape) == 3:
            a4c_seq = a4c_seq[..., np.newaxis] * np.ones((1, 1, 1, 3))

        a4c_seq = (a4c_seq - self.mean) / self.std

        # 使用填充函数填充到128
        a4c_seq = self.pad_to_shape(a4c_seq)
        a4c_gt = self.pad_to_shape(a4c_gt)

        if self.n_frames > a4c_seq.shape[0]:
            # TYPE 1: MIRRORING
            # a4c_seq = expand_and_mirror(a4c_seq, self.n_frames)
            # tmp = np.zeros_like(a4c_seq)[...,0]
            # tmp[:a4c_gt.shape[0]] = a4c_gt
            # a4c_gt = tmp.copy()

            # TYPE 2: PADDING TO ZERO
            # tmp = np.zeros((self.n_frames, 112, 112, 3)).astype(a4c_seq.dtype)
            # tmp[:a4c_seq.shape[0]] = a4c_seq
            # a4c_seq = tmp.copy()
            # tmp = np.zeros_like(a4c_seq)[...,0]
            # tmp[:a4c_gt.shape[0]] = a4c_gt
            # a4c_gt = tmp.copy()

            # TYPE 3:
            # a4c_seq = pad_array(a4c_seq, self.n_frames)
            # a4c_gt  = pad_array(a4c_gt,  self.n_frames)

            # TYPE 4:
            # a4c_seq = pad_array_with_images(a4c_seq, self.n_frames)
            # a4c_gt  = pad_array(a4c_gt,  self.n_frames)

            # TYPE 5:
            # a4c_seq = pad_array_with_origin_images_seq(a4c_seq, self.n_frames)
            # a4c_gt = pad_array_with_origin_images_gt(a4c_gt, self.n_frames)

            # TYPE 6:
            valid_frame_num = np.sum(a4c_seq.sum(1).sum(-1).sum(-1) != 0)
            a4c_seq = valid_crop_resize(a4c_seq, valid_frame_num, [1], self.n_frames)
            a4c_gt = a4c_gt[:, :, :, None]
            a4c_gt = valid_crop_resize(a4c_gt, valid_frame_num, [1], self.n_frames)[..., 0]  # [...，0]中..表示保留了前面的所有维度，0表示选择最后一维的第一个元荼

        else:
            valid_frame_num = self.n_frames

        assert a4c_seq.shape[0] == self.n_frames

        # (F, H, W, C) --> (C, H, W, F)
        a4c_seq = a4c_seq.transpose((3, 1, 2, 0))
        a4c_gt = a4c_gt.transpose((1, 2, 0))

        a4c_seq = np.float32(a4c_seq)
        a4c_gt = np.float32(a4c_gt)

        # 获取ED和ES的标注
        # a4c_gt = {
        #     'label_index': [0, int(valid_frame_num - 1)],
        #     'video_gt': a4c_gt,
        # }

        # all
        a4c_gt = {
            'label_index': [0, self.n_frames - 1],
            'video_gt': a4c_gt,
        }

        # 实际视频数据 标注数据 病人序号
        return a4c_seq, a4c_gt, patient




