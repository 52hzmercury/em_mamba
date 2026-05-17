import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(
            self,
            loss_type='dice'
    ):
        super().__init__()

        if not isinstance(loss_type, list):
            loss_type = [loss_type]

        # 检查不支持的 loss type
        for l in loss_type:
            if l not in ['bce', 'sbce', 'dice', 'mse', 'hccmse', 'hccdice', 'focal',
                         'tversky', 'jaccard', 'onlyhcc', 'dice+jaccard', 'dice+focal', 'jaccard+focal', 'dice+jaccard+focal', 'dice+chamfer+focal',
                         'dice+jaccard+tversky+focal']:
                raise ValueError(f'Loss type {l} is not recognized ...')

        # --- 初始化基础损失函数 ---
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.7, beta=0.3)
        self.jaccard_loss = smp.losses.JaccardLoss(mode='binary')
        self.sbce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)
        self.mse_loss = nn.MSELoss()

        self.loss_type = loss_type

        # --- 【关键修改】初始化自适应权重参数 ---
        if 'dice+jaccard+focal' in self.loss_type:
            # 使用对数初始化，确保 Softmax 后精确等于 0.4, 0.3, 0.3
            # ln(0.4) ≈ -0.9163, ln(0.3) ≈ -1.2040
            init_weights = torch.tensor([0.4, 0.3, 0.3])
            self.djf_weights = nn.Parameter(torch.log(init_weights))


        if 'dice+jaccard' in self.loss_type:
            self.djf_weights = nn.Parameter(torch.zeros(2))

        if 'dice+focal' in self.loss_type:
            self.djf_weights = nn.Parameter(torch.zeros(2))

        if 'jaccard+focal' in self.loss_type:
            self.djf_weights = nn.Parameter(torch.zeros(2))

    def pearson(self, preds, labels):
        """
        计算皮尔逊相关系数损失
        """
        preds_mean = preds.mean(dim=1)
        labels_mean = labels.mean(dim=1)

        preds_cycle = preds_mean.std(dim=(1, 2))
        labels_cycle = labels_mean.std(dim=(1, 2))

        signal1_centered = preds_cycle - preds_cycle.mean()
        signal2_centered = labels_cycle - labels_cycle.mean()

        numerator = torch.sum(signal1_centered * signal2_centered)
        denominator = torch.sqrt(torch.sum(signal1_centered ** 2) * torch.sum(signal2_centered ** 2))
        correlation = numerator / (denominator + 1e-8)

        loss = F.mse_loss(correlation, torch.tensor(1.0).to(correlation.device))
        return loss

    # 如果你需要 spearman，请确保类里定义了该方法，或者像 pearson 一样实现
    # def spearman(self, preds, labels): ...

    def forward(self, preds, labels):
        loss = 0.

        # ... 其他 loss 的处理保持不变 ...
        if 'bce' in self.loss_type:
            loss += self.bce_loss(preds, labels)
        if 'dice' in self.loss_type:
            loss += self.dice_loss(preds, labels)
        if 'focal' in self.loss_type:
            loss += self.focal_loss(preds, labels)
        if 'tversky' in self.loss_type:
            loss += self.tversky_loss(preds, labels)
        if 'sbce' in self.loss_type:
            loss += self.sbce(preds, labels)
        if 'mse' in self.loss_type:
            loss += self.mse_loss(preds, labels)
        if 'jaccard' in self.loss_type:
            loss += self.jaccard_loss(preds, labels)

        if 'hccmse' in self.loss_type:
            loss += (self.mse_loss(preds, labels) + 0.1 * self.mse_loss(preds, labels))

        if 'hccdice' in self.loss_type:
            loss += (self.dice_loss(preds, labels) + 0.1 * self.pearson(preds, labels))

        if 'onlyhcc' in self.loss_type:
            # 注意：确保 self.spearman 存在
            # loss += self.spearman(preds, labels)
            pass

        if 'dice+jaccard' in self.loss_type:
            # 1. 对可学习参数进行 Softmax，保证权重之和为1且非负
            weights = F.softmax(self.djf_weights, dim=0)

            # 2. 分别计算损失
            l_dice = self.dice_loss(preds, labels)
            l_jacc = self.jaccard_loss(preds, labels)

            # 3. 加权求和
            loss += weights[0] * l_dice + weights[1] * l_jacc
            print(f'Dice: {weights[0]:.4f}, Jaccard: {weights[1]:.4f}')

        if 'dice+focal' in self.loss_type:
            # 1. 对可学习参数进行 Softmax，保证权重之和为1且非负
            weights = F.softmax(self.djf_weights, dim=0)

            # 2. 分别计算损失
            l_dice = self.dice_loss(preds, labels)
            l_focal = self.focal_loss(preds, labels)

            # 3. 加权求和
            loss += weights[0] * l_dice + weights[1] * l_focal
            print(f'Dice: {weights[0]:.4f}, Focal: {weights[1]:.4f}')

        if 'jaccard+focal' in self.loss_type:
            # 1. 对可学习参数进行 Softmax，保证权重之和为1且非负
            weights = F.softmax(self.djf_weights, dim=0)

            # 2. 分别计算损失
            l_dice = self.dice_loss(preds, labels)
            l_jacc = self.jaccard_loss(preds, labels)

            # 3. 加权求和
            loss += weights[0] * l_dice + weights[1] * l_jacc
            print(f'Focal: {weights[0]:.4f}, Jaccard: {weights[1]:.4f}')

        # --- 【关键修改】自适应权重计算逻辑 ---
        if 'dice+jaccard+focal' in self.loss_type:
            # 1. 对可学习参数进行 Softmax，保证权重之和为1且非负
            weights = F.softmax(self.djf_weights, dim=0)

            # 2. 分别计算损失
            l_dice = self.dice_loss(preds, labels)
            l_jacc = self.jaccard_loss(preds, labels)
            l_focal = self.focal_loss(preds, labels)

            # 3. 加权求和 (weights[0]对应Dice, [1]对应Jaccard, [2]对应Focal)
            # loss += weights[0] * l_dice + weights[1] * l_jacc + weights[2] * l_focal

            # print(f'Dice: {weights[0]:.4f}, Jaccard: {weights[1]:.4f}, Focal: {weights[2]:.4f}')
            loss += 0.4692 * l_dice + 0.1687 * l_jacc + 0.3620 * l_focal

        if 'dice+jaccard+tversky+focal' in self.loss_type:
            loss += 0.3 * self.dice_loss(preds, labels) + 0.1 * self.jaccard_loss(preds, labels) + 0.3 * self.tversky_loss(
                preds, labels) + 0.3 * self.focal_loss(preds, labels)

        return loss