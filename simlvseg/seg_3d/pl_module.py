import torch

from ..pl_module import SegModule

class Seg3DModule(SegModule):
    def preprocess_batch_imgs(self, imgs):
        return imgs
    
    def postprocess_batch_preds_and_targets(self, preds, targets):
        out_preds  = []
        out_labels = []
        # origin  filename
        if len(preds) != len(targets['filename']):
            raise ValueError("The number of predictions and the number of targets are different ...")

        for i in range(len(preds)):
            pred = preds[i]

            trace_mask = targets['trace_mask'][i][None, :]
            pred_trace = pred[..., targets['rel_trace_index'][i]]

            out_preds.extend([pred_trace[None, :]])
            out_labels.extend([trace_mask])

        out_preds  = torch.cat(out_preds)
        out_labels = torch.cat(out_labels)

        return out_preds, out_labels


    # 训练用弱监督函数
    def postprocess_batch_preds_and_targets_camus(self, preds, targets):


        # 全监督
        out_preds = preds
        out_labels = targets['video_gt'][:, None, :]  # None表示在第0维之后插入一个维度

        #                  [batch_size channel height width depth]
        # preds.shape      [4, 1, 112, 112, 128]
        # out_preds.shape  [4, 1, 112, 112, 128]
        # targets.shape    [4, 112, 112, 128]
        # out_labels.shape [4, 1, 112, 112, 128]

        # 弱监督  只提取两帧
        # out_preds = []
        # out_labels = []
        # preds = preds
        # targets['video_gt'] = targets['video_gt'][:, None, :]  # None表示在第0维之后插入一个维度
        # for i in range(len(preds)):
        #     lable_index = [targets['label_index'][0][i], targets['label_index'][1][i]]
        #     valid_preds = preds[i][..., lable_index]
        #     valid_labels = targets['video_gt'][i][..., lable_index]
        #
        #     out_preds.extend([valid_preds])
        #     out_labels.extend([valid_labels])
        #
        # out_preds = torch.cat(out_preds)
        # out_labels = torch.cat(out_labels)

        return out_preds, out_labels


    # 验证用提取全部数据函数
    def postprocess_batch_preds_and_targets_camus_test(self, preds, targets):

        # 全监督
        out_preds = preds[0]
        out_labels = targets['video_gt'][:, None, :][0]  # None表示在第0维之后插入一个维度

        return out_preds, out_labels


