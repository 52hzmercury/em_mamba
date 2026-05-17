import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将父目录加入路径，确保能导入 simlvseg 包
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

# 导入 medpy 计算指标
try:
    from medpy.metric import binary
except ImportError:
    print("Error: medpy is not installed. Please install it using 'pip install medpy'")
    sys.exit(1)

from simlvseg.seg_3d.dataset import Seg3DDataset  #
from simlvseg.seg_3d.pl_module import Seg3DModule  #
from simlvseg.seg_3d.preprocessing import get_preprocessing_for_training  #
from simlvseg.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="3D Segmentation Testing with Bootstrap CI")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, help="Path to the dataset", required=True)
    parser.add_argument('--checkpoint_path', type=str, help="Path to the trained .ckpt file", required=True)

    # 保持与训练配置一致
    parser.add_argument('--frames', type=int, default=32, help="Number of frames")
    parser.add_argument('--period', type=int, default=1, help="Period")
    parser.add_argument('--mean', type=float, nargs=3, default=(0.12741163, 0.1279413, 0.12912785))
    parser.add_argument('--std', type=float, nargs=3, default=(0.19557191, 0.19562256, 0.1965878))

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for testing")
    parser.add_argument('--encoder', type=str, default='3d_unet', help="Encoder type")

    # Bootstrap 参数
    parser.add_argument('--n_bootstraps', type=int, default=10000, help="Number of resamples")

    args = parser.parse_args()
    return args


def calculate_bootstrap_ci(data, n_bootstraps=10000, alpha=0.05):
    """
    使用 Bootstrap 方法计算 95% 置信区间
    """
    data = np.array(data)
    # 移除 NaN 值
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return 0.0, 0.0, 0.0

    n = len(data)
    means = []

    # 重采样循环
    for _ in range(n_bootstraps):
        # 有放回采样
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.sort(means)

    # 计算百分位点
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)
    original_mean = np.mean(data)

    return original_mean, lower_bound, upper_bound


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. 预处理配置
    preprocessing = get_preprocessing_for_training(
        args.frames, args.mean, args.std
    )

    # 2. 数据集与 DataLoader (与 train 脚本保持一致)
    print('Configuring test dataset ...')
    test_dataset = Seg3DDataset(
        args.data_path,
        "test",
        args.frames,
        args.period,
        False,
        preprocessing,
        None,
        test=True  #
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # 3. 加载模型
    print(f'Loading model from {args.checkpoint_path} ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seg3DModule.load_from_checkpoint(
        args.checkpoint_path,
        encoder_name=args.encoder,
        strict=False
    )
    model.to(device)
    model.eval()

    # 存储每个样本的指标
    metrics_log = {'dsc': [], 'iou': [], 'hd95': [], 'assd': []}

    print('Starting inference ...')
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs, targets = batch

            # 将 Tensor 移动到 GPU
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device)

            # 预处理
            imgs = model.preprocess_batch_imgs(imgs)  #
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.to(device)
            elif isinstance(imgs, (list, tuple)):
                imgs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in imgs]

            # 推理
            preds_raw = model(imgs)

            # 后处理：关键步骤，裁剪和对齐 Trace Frame
            preds, labels = model.postprocess_batch_preds_and_targets(preds_raw, targets)  #

            # 二值化
            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > 0.5).float()

            preds_np = preds_bin.cpu().numpy().astype(np.uint8)
            labels_np = labels.cpu().numpy().astype(np.uint8)

            # 计算 Batch 中每个样本的指标
            for i in range(preds_np.shape[0]):
                p = preds_np[i, 0, :, :]
                g = labels_np[i, 0, :, :]

                # 处理空 Mask 情况
                if g.sum() == 0:
                    if p.sum() == 0:  # 双空，完美
                        metrics_log['dsc'].append(1.0)
                        metrics_log['iou'].append(1.0)
                        metrics_log['hd95'].append(0.0)
                        metrics_log['assd'].append(0.0)
                    else:  # 假阳性
                        metrics_log['dsc'].append(0.0)
                        metrics_log['iou'].append(0.0)
                        metrics_log['hd95'].append(np.nan)
                        metrics_log['assd'].append(np.nan)
                    continue

                if p.sum() == 0:  # 假阴性
                    metrics_log['dsc'].append(0.0)
                    metrics_log['iou'].append(0.0)
                    metrics_log['hd95'].append(np.nan)
                    metrics_log['assd'].append(np.nan)
                    continue

                # Medpy 计算
                metrics_log['dsc'].append(binary.dc(p, g))
                metrics_log['iou'].append(binary.jc(p, g))
                metrics_log['hd95'].append(binary.hd95(p, g))
                metrics_log['assd'].append(binary.assd(p, g))

    # 4. Bootstrap 统计输出
    print("\n" + "=" * 80)
    print(f"Test Results with Bootstrap 95% CI (Resamples={args.n_bootstraps})")
    print("=" * 80)
    print(f"{'Metric':<10} | {'Mean':<10} | {'95% CI (Lower - Upper)':<30}")
    print("-" * 60)

    results_summary = {}

    for key in ['dsc', 'iou', 'hd95', 'assd']:
        mean_val, lower, upper = calculate_bootstrap_ci(metrics_log[key], n_bootstraps=args.n_bootstraps)
        results_summary[key] = (mean_val, lower, upper)

        print(f"{key.upper():<10} | {mean_val:.4f}     | [{lower:.4f} - {upper:.4f}]")

    print("=" * 80)


if __name__ == '__main__':
    main()