import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import torch

from .model import get_model
from .loss import SegLoss
from .utils import get_crop_from_coors
import torch, gc


class BaseModule(pl.LightningModule):
    def preprocess_batch_imgs(self, imgs):
        raise NotImplementedError

    def postprocess_batch_preds_and_targets(self, preds, targets):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def postprocess_batch_preds_and_targets_camus(self, preds, labels):
        raise NotImplementedError

    def calculate_metrics(self, set_name, preds, labels):
        # Calculate the metrics
        metrics = [[name, fn(preds, labels)] for name, fn in self.metrics.items()]
        # Print the metrics on the terminal
        for name, value in metrics:
            self.log(f"{set_name}_{name}", value, prog_bar=True, logger=True)

    def calculate_metrics_batch(self, preds, labels):
        # Calculate the metrics
        metrics = [[name, fn(preds, labels)] for name, fn in self.metrics.items()]

        return metrics

    def val_test_epoch_end(self, set_name, step_outputs):
        preds = []
        labels = []

        for output in step_outputs:
            preds.append(output['batch_preds'])
            labels.append(output['batch_labels'])

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        loss = self.criterion(preds, labels)

        self.log(f"{set_name}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.calculate_metrics(set_name, preds, labels)

        # camus
        # losses = []
        # value_dsc=[]
        # value_iou=[]
        # value_dice_loss=[]
        #
        # for output in step_outputs:
        #     batch_pred = output['batch_preds']
        #     batch_labels = output['batch_labels']
        #
        #     loss = self.criterion(batch_pred, batch_labels)
        #     losses.append(loss)
        #
        #     metrics = self.calculate_metrics_batch(batch_pred, batch_labels)
        #     # metrics[0][1] value_dsc
        #     value_dsc.append(metrics[0][1])
        #     # metrics[1][1] value_iou
        #     value_iou.append(metrics[1][1])
        #     # metrics[2][1] value_dice_loss
        #     value_dice_loss.append(metrics[2][1])
        #
        # # 计算所有 loss 的平均值
        # loss_mean = sum(losses) / len(losses)
        # self.log(f"{set_name}_loss", loss_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # # 计算所有 dsc 的平均值
        # avg_dsc = sum(value_dsc) / len(value_dsc) if value_dsc else 0
        # self.log(f"{set_name}_dsc", avg_dsc, prog_bar=True, logger=True)
        # # 计算所有 iou 的平均值
        # avg_iou = sum(value_iou) / len(value_iou) if value_iou else 0
        # self.log(f"{set_name}_iou", avg_iou, prog_bar=True, logger=True)
        # # 计算所有 dice_loss 的平均值
        # avg_dice_loss = sum(value_dice_loss) / len(value_dice_loss) if value_dice_loss else 0
        # self.log(f"{set_name}_dice_loss", avg_dice_loss, prog_bar=True, logger=True)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        imgs = self.preprocess_batch_imgs(imgs)

        preds = self.forward(imgs)

        # preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        preds, labels = self.postprocess_batch_preds_and_targets_camus(preds, targets)


        loss = self.criterion(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.calculate_metrics('train', preds, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        imgs = self.preprocess_batch_imgs(imgs)

        preds = self.forward(imgs)

        # preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        preds, labels = self.postprocess_batch_preds_and_targets_camus(preds, targets)
        # preds, labels = self.postprocess_batch_preds_and_targets_camus_val(preds, targets)

        return {'batch_preds': preds, 'batch_labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        return self.val_test_epoch_end('val', validation_step_outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.val_test_epoch_end('test', test_step_outputs)


class SegModule(BaseModule):
    def __init__(
            self,
            encoder_name,
            weights=None,
            pretrained_type='encoder',
            img_size=None,
            # loss_type='dice',
            # loss_type='jaccard',
            # loss_type='hccdie',
            # loss_type='hccmse',
            # loss_type='tversky',
            # loss_type='focal',
            loss_type='dice+jaccard',
            # loss_type='dice+focal',
            # loss_type='jaccard+focal',
            # loss_type='dice+jaccard+focal',
    ):
        super().__init__()

        self.model = get_model(encoder_name, weights, pretrained_type, img_size)

        self.criterion = SegLoss(loss_type)
        self.metrics = {
            'dsc': smp_utils.metrics.Fscore(activation='sigmoid'),
            'iou': smp.utils.metrics.IoU(activation='sigmoid'),
            'dice_loss': smp.losses.DiceLoss(mode='binary', from_logits=True),
        }

    def preprocess_batch_imgs(self, imgs):

        # batch_imgs preprocessing for super images
        super_images, videos = imgs
        return super_images

    def postprocess_batch_preds_and_targets(self, preds, targets):

        out_preds = []
        out_labels = []

        if len(preds) != len(targets['filename']):
            raise ValueError("The number of predictions and the number of targets are different ...")

        for i in range(len(preds)):
            pred = preds[i]

            trace_mask = targets['trace_mask'][i][None, :]
            pos_trace_frame = self.__get_pos_frame(targets['pos_trace_frame'], i)

            # Change from the channel-first into the channel-last format
            pred = pred.permute((1, 2, 0))

            pred_trace = get_crop_from_coors(pred, pos_trace_frame)

            # Change from the channel-last into the channel-first format
            pred_trace = pred_trace.permute((2, 0, 1))

            out_preds.extend([pred_trace[None, :]])
            out_labels.extend([trace_mask])
        # origin
        out_preds = torch.cat(out_preds)[:, :, :112, :112].contiguous()
        out_labels = torch.cat(out_labels)[:, :, :112, :112].contiguous()

        return out_preds, out_labels

    @staticmethod
    def __get_pos_frame(tensor_pos_frame, index):

        """
        Convert from
        [
            [
                tensor([224,   0, 112, 224, 112, 112, 224, 336,   0, 112,   0, 224, 336,   0,112, 112], device='cuda:0'),
                tensor([560,   0,   0, 336, 224,   0, 336,   0, 224, 336, 336, 560,   0, 560, 336, 224], device='cuda:0')
            ],
            [
                tensor([336, 112, 224, 336, 224, 224, 336, 448, 112, 224, 112, 336, 448, 112, 224, 224], device='cuda:0'),
                tensor([672, 112, 112, 448, 336, 112, 448, 112, 336, 448, 448, 672, 112, 672, 448, 336], device='cuda:0')
            ]
        ]

        To (for index=0) --> [[224, 560], [336, 672]]
        """

        tl = tensor_pos_frame[0]
        br = tensor_pos_frame[1]

        return [[tl[0][index], tl[1][index]],
                [br[0][index], br[1][index]]]

    def configure_optimizers(self):
        # AdamW
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=3e-4,
            weight_decay=1e-5, amsgrad=True,
        )

        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=1e-5,
        #     weight_decay=1e-5, amsgrad=True,
        # )

        # SGD
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=0.1,  # 设置学习率
        #     momentum=0.9,  # 设置动量
        #     weight_decay=1e-5  # 设置权重衰减
        # )

        # MultiStepLR
        # 默认
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
          optimizer, milestones=[45, 60], gamma=0.1,
        )

        # 35 50
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[35, 50], gamma=0.1,
        # )

        # 定义LinearLR调度器，线性增加
        # scheduler = LinearLR(
        #     optimizer,
        #     end_lr=1e-2,  # 最终学习率
        #     num_iter=60,  # 总迭代次数
        # )

        # 定义ExponentialLR调度器
        # scheduler = ExponentialLR(
        #     optimizer,
        #     end_lr=1e-2,  # 最终学习率
        #     num_iter=60,  # 总迭代次数
        # )

        # 定义WarmupCosineSchedule调度器
        # scheduler = WarmupCosineSchedule(
        #     optimizer,
        #     warmup_steps=5,  # 线性warmup步骤
        #     t_total=60,  # 总训练步骤
        #     cycles=0.5,  # 余弦周期
        # )

        # 定义LinearWarmupCosineAnnealingLR调度器
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=5,  # 热身epoch数
        #     max_epochs=60,  # 总训练epoch数
        #     warmup_start_lr=1e-6,  # 热身开始的学习率
        #     eta_min=1e-6,  # 最小学习率
        # )

        return [optimizer], [scheduler]
