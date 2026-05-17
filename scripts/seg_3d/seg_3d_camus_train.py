import os
import sys



sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import argparse
import numpy as np
import pytorch_lightning as pl
import random
import torch

from torch.utils.data import DataLoader

from simlvseg.augmentation import get_augmentation
from simlvseg.utils import set_seed
from simlvseg.seg_3d.dataset import Seg3DDatasetCamus
from simlvseg.seg_3d.pl_module import Seg3DModule
from simlvseg.seg_3d.preprocessing import get_preprocessing_for_training

def parse_args():
    parser = argparse.ArgumentParser(description="Weakly Supervised Video Segmentation Training with 3D Models")

    parser.add_argument('--seed', type=int, default=42)

    # Paths and dataset related arguments
    parser.add_argument('--data_path', type=str, help="Path to the dataset", required=True)
    parser.add_argument('--mean', type=float, nargs=3, default=(0.12741163, 0.1279413, 0.12912785),
                        help="Mean normalization value (can be a list or tuple)")
    parser.add_argument('--std', type=float, nargs=3, default=(0.19557191, 0.19562256, 0.1965878),
                        help="Standard deviation normalization value (can be a list or tuple)")

    # Model and training related arguments
    parser.add_argument('--encoder', type=str, default='3d_unet', help="Encoder type")
    parser.add_argument('--frames', type=int, default=32, help="Number of frames")
    parser.add_argument('--period', type=int, default=1, help="Period")
    parser.add_argument('--pct_train', type=float, default=None, help="Percentage of training data to use (can be None or a float)")

    # DataLoader arguments
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation")

    # Training procedure arguments
    parser.add_argument('--epochs', type=int, default=70, help="Number of epochs to train for")
    parser.add_argument('--val_check_interval', type=float, default=0.25, help="Interval at which to check validation performance")

    # Checkpointing arguments
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint file (can be None or a string)")
    parser.add_argument('--pretrained_type', type=str, default='pl_full', choices=['pl_full', 'encoder'],
                        help="Type of pretraining to use ('pl_full' or 'encoder')")


    args = parser.parse_args()
    return args
def read_patient_names(file_path):
    with open(file_path, 'r') as file:
        patient_names = [line.strip() for line in file.readlines()]
    return patient_names

class DataModule(pl.LightningDataModule):
    def __init__(self, augmentation, preprocessing):
        super().__init__()

        # print('Configuring dataset ...')
        # self.dataset = Seg3DDatasetCamus(
        #     args.data_path,
        #     args.frames,
        #     args.mean,
        #     args.std
        # )
        #
        # # 将数据集分割成训练集、验证集和测试集
        # train_size = int(0.7 * len(self.dataset))
        # valid_size = int(0.1 * len(self.dataset))
        # test_size = len(self.dataset) - train_size - valid_size
        #
        # self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
        #     self.dataset, [train_size, valid_size, test_size]
        # )


        # 读取文件内容

        # SAM划分
        train_patient_names = read_patient_names('scripts/camus/database_split/camus_train_filenames.txt')
        val_patient_names = read_patient_names('scripts/camus/database_split/camus_val_filenames.txt')
        test_patient_names = read_patient_names('scripts/camus/database_split/camus_test_filenames.txt')

        # CAMUS官方划分
        # train_patient_names = read_patient_names('scripts/camus/database_split/subgroup_training.txt')
        # val_patient_names = read_patient_names('scripts/camus/database_split/subgroup_validation.txt')
        # test_patient_names = read_patient_names('scripts/camus/database_split/subgroup_testing.txt')

        print('Configuring train dataset ...')
        # 训练数据集
        self.train_dataset = Seg3DDatasetCamus(
            args.data_path,
            args.frames,
            args.mean,
            args.std,
            train_patient_names
        )

        print('Configuring val dataset ...')
        # 验证数据集
        self.val_dataset = Seg3DDatasetCamus(
            args.data_path,
            args.frames,
            args.mean,
            args.std,
            val_patient_names
        )

        print('Configuring test dataset ...')
        # 测试数据集
        self.test_dataset = Seg3DDatasetCamus(
            args.data_path,
            args.frames,
            args.mean,
            args.std,
            test_patient_names
        )



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)

    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, drop_last=False)

    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, drop_last=False)

if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    augmentation = get_augmentation(args.frames)

    preprocessing = get_preprocessing_for_training(
        args.frames,
        args.mean,
        args.std,
    )

    model = Seg3DModule(args.encoder, args.checkpoint, args.pretrained_type)

    # dm = DataModule(augmentation, preprocessing)
    dm = DataModule(None,None)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        mode='max',
        monitor='val_dsc',
        verbose=True,
        save_last=False,

        # 保存最佳
        save_top_k=1,
        save_weights_only=False,
    )

    # 32位精度
    # 多卡并行变为ddp后，需要调高lr，倍数为对应显卡张数或根号倍
    # 多卡 devices=[0, 1], strategy="ddp", sync_batchnorm=True,
    # 单卡 devices=[0]
    trainer = pl.Trainer(accelerator="gpu", devices=[0, 1, 3], strategy="ddp", sync_batchnorm=True, max_epochs=args.epochs,
                        val_check_interval=args.val_check_interval,
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback])

    # trainer = pl.Trainer(accelerator="gpu", devices=[0, 1], strategy="ddp", sync_batchnorm=True, max_epochs=args.epochs,
    #                      val_check_interval=args.val_check_interval,
    #                      log_every_n_steps=10,
    #                      callbacks=[checkpoint_callback, pl.callbacks.EarlyStopping(monitor='val_dsc', mode='max', patience=30)])


    trainer.fit(model, dm)
    trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path='best')

