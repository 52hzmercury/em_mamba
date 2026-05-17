import argparse
import pytorch_lightning as pl
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import math
import yaml
from skimage import measure
from sklearn.metrics import confusion_matrix

import collections
import skimage.draw
import torchvision
import cv2
import pandas
from medpy.metric.binary import assd as medpy_assd
from medpy.metric.binary import hd95 as medpy_hd95


from simlvseg.utils import defaultdict_of_lists


class Echo(torchvision.datasets.VisionDataset):

    def __init__(
            self,
            root,
            pred_dir=None,
            split="test",
            target_type="EF",
            external_test_location=None,
            frame_shape=(128, 128)
    ):
        super().__init__(root)

        self.frame_shape = frame_shape
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.pred_dir = pred_dir
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []


        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]
            self.outcome = data.values.tolist()

            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(defaultdict_of_lists)


            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                if(header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]):
                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))
                if (header == ["FileName", "X", "Y", "Frame"]):
                    for line in f:
                        filename, x, y, frame = line.strip().split(',')
                        x = float(x)
                        y = float(y)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x, y))


            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])


            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]


    def __getitem__(self, index):

        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        target = []

        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(int(self.frames[key][0]))
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]





                if t.shape[1] == 4:
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))
                else:
                    assert t.shape[1] == 2
                    x, y = t[:, 0], t[:, 1]

                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), self.frame_shape)
                mask = np.zeros(self.frame_shape, np.float32)
                mask[r, c] = 1
                target.append(mask)
            elif t in ["LargePred", "SmallPred"]:
                if t == "LargePred":
                    mask = self.get_pred(key, self.frames[key][-1], phase='esv')
                else:
                    mask = self.get_pred(key, self.frames[key][0], phase='edv')
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))


        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]

        return target


    def __len__(self):
        return len(self.fnames)


    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


    def get_pred(self, video_name, frame, phase='edv'):
        if self.pred_dir is None:
            return None
        else:
            filename = os.path.join(self.pred_dir,
                                    "{}_{}_{}.png".format(video_name.strip('.avi'), frame, phase))
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img.astype(np.float32) / 255.0



class TestData():

    def mean_and_std(self, values):
        mean = np.mean(values)
        std = np.std(values)
        return mean, std


    def compute_ci(self, values, num_bootstraps=10000):


        values = np.array(values)
        n = len(values)
        if n < 2:
            return np.mean(values), np.mean(values)

        bootstrapped_means = []



        for _ in range(num_bootstraps):

            sample = np.random.choice(values, size=n, replace=True)
            bootstrapped_means.append(np.mean(sample))

        sorted_means = np.sort(bootstrapped_means)

        lower = sorted_means[int(0.025 * len(sorted_means))]
        upper = sorted_means[int(0.975 * len(sorted_means))]

        return lower, upper


    def run_test(self, data_dir, output_dir, pred_dir, num_workers=1):
        os.makedirs(output_dir, exist_ok=True)

        for split in ["test"]:
            dataset = Echo(root=data_dir, pred_dir=pred_dir, split=split,
                           target_type=["LargeTrace", "SmallTrace", "LargePred", "SmallPred"])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers,
                                                     shuffle=False, drop_last=False)


            assd, hd95, sen, large_inter, large_union, small_inter, small_union = self.run_epoch(dataloader)



            overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter + 1e-7)
            large_dice = 2 * large_inter / (large_union + large_inter + 1e-7)
            small_dice = 2 * small_inter / (small_union + small_inter + 1e-7)










            overall_iou = (large_inter + small_inter) / (large_union + small_union + 1e-7)
            large_iou = large_inter / (large_union + 1e-7)
            small_iou = small_inter / (small_union + 1e-7)




            def calc_stats(data):
                mean, std = self.mean_and_std(data)
                ci_low, ci_high = self.compute_ci(data)
                return mean, std, ci_low, ci_high


            hd95_mean, hd95_std, hd95_low, hd95_high = calc_stats(hd95)
            assd_mean, assd_std, assd_low, assd_high = calc_stats(assd)
            sen_mean, sen_std, sen_low, sen_high = calc_stats(sen)

            ov_dice_m, ov_dice_s, ov_dice_l, ov_dice_h = calc_stats(overall_dice)
            lg_dice_m, lg_dice_s, lg_dice_l, lg_dice_h = calc_stats(large_dice)
            sm_dice_m, sm_dice_s, sm_dice_l, sm_dice_h = calc_stats(small_dice)

            ov_iou_m, ov_iou_s, ov_iou_l, ov_iou_h = calc_stats(overall_iou)
            lg_iou_m, lg_iou_s, lg_iou_l, lg_iou_h = calc_stats(large_iou)
            sm_iou_m, sm_iou_s, sm_iou_l, sm_iou_h = calc_stats(small_iou)




            with open(os.path.join(output_dir, "{}_assd.csv".format(split)), "w") as f:
                f.write("Filename, ASSD\n")
                for filename, val in zip(dataset.fnames, assd):
                    f.write("{}, {}\n".format(filename, val))
                f.write("Mean ASSD, {}\n".format(assd_mean))
                f.write("Std ASSD, {}\n".format(assd_std))
                f.write("95% CI Lower, {}\n".format(assd_low))
                f.write("95% CI Upper, {}\n".format(assd_high))


            with open(os.path.join(output_dir, "{}_hd95.csv".format(split)), "w") as f:
                f.write("Filename, HD95\n")
                for filename, val in zip(dataset.fnames, hd95):
                    f.write("{}, {}\n".format(filename, val))
                f.write("Mean HD95, {}\n".format(hd95_mean))
                f.write("Std HD95, {}\n".format(hd95_std))
                f.write("95% CI Lower, {}\n".format(hd95_low))
                f.write("95% CI Upper, {}\n".format(hd95_high))


            with open(os.path.join(output_dir, "{}_sen.csv".format(split)), "w") as f:
                f.write("Filename, SEN\n")
                for filename, val in zip(dataset.fnames, sen):
                    f.write("{}, {}\n".format(filename, val))
                f.write("Mean SEN, {}\n".format(sen_mean))
                f.write("Std SEN, {}\n".format(sen_std))
                f.write("95% CI Lower, {}\n".format(sen_low))
                f.write("95% CI Upper, {}\n".format(sen_high))


            with open(os.path.join(output_dir, "{}_dice.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                    g.write("{},{},{},{}\n".format(filename, overall, large, small))

                g.write("Metric, Mean, Std, 95% CI Lower, 95% CI Upper\n")
                g.write("Overall Dice, {}, {}, {}, {}\n".format(ov_dice_m, ov_dice_s, ov_dice_l, ov_dice_h))
                g.write("Large Dice, {}, {}, {}, {}\n".format(lg_dice_m, lg_dice_s, lg_dice_l, lg_dice_h))
                g.write("Small Dice, {}, {}, {}, {}\n".format(sm_dice_m, sm_dice_s, sm_dice_l, sm_dice_h))


            with open(os.path.join(output_dir, "{}_iou.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, overall, large, small) in zip(dataset.fnames, overall_iou, large_iou, small_iou):
                    g.write("{},{},{},{}\n".format(filename, overall, large, small))

                g.write("Metric, Mean, Std, 95% CI Lower, 95% CI Upper\n")
                g.write("Overall IOU, {}, {}, {}, {}\n".format(ov_iou_m, ov_iou_s, ov_iou_l, ov_iou_h))
                g.write("Large IOU, {}, {}, {}, {}\n".format(lg_iou_m, lg_iou_s, lg_iou_l, lg_iou_h))
                g.write("Small IOU, {}, {}, {}, {}\n".format(sm_iou_m, sm_iou_s, sm_iou_l, sm_iou_h))


            with open(os.path.join(output_dir, "log.csv"), "w") as f:
                f.write("Metric, Mean, Std, 95% CI Lower, 95% CI Upper, 95%CI dis\n")


                f.write(f"{split} Dice,{ov_dice_m:.4f},{ov_dice_s:.4f},{ov_dice_l:.4f},{ov_dice_h:.4f},{(ov_dice_h - ov_dice_l) / 2:.4f}\n")




                f.write(f"{split} IOU,{ov_iou_m:.4f},{ov_iou_s:.4f},{ov_iou_l:.4f},{ov_iou_h:.4f},{(ov_iou_h - ov_iou_l) / 2:.4f}\n")


                f.write(f"{split} ASSD,{assd_mean:.4f},{assd_std:.4f},{assd_low:.4f},{assd_high:.4f}\n")
                f.write(f"{split} HD95,{hd95_mean:.4f},{hd95_std:.4f},{hd95_low:.4f},{hd95_high:.4f}\n")


                f.flush()


    def run_epoch(self, dataloader):

        assd_list = []
        hd95_list = []
        sen_list = []

        large_inter = 0
        large_union = 0
        small_inter = 0
        small_union = 0
        large_inter_list = []
        large_union_list = []
        small_inter_list = []
        small_union_list = []

        for large_trace, small_trace, large_pred, small_pred in tqdm.tqdm(dataloader):
            assd_large = self.compute_assd(large_trace, large_pred)
            assd_small = self.compute_assd(small_trace, small_pred)
            hd95_large = self.compute_hd95(large_trace, large_pred)
            hd95_small = self.compute_hd95(small_trace, small_pred)
            sen_large = self.compute_sen(large_trace, large_pred)
            sen_small = self.compute_sen(small_trace, small_pred)

            assd = (assd_large + assd_small) / 2
            hd95 = (hd95_large + hd95_small) / 2
            sen = (sen_large + sen_small) / 2

            assd_list.append(assd)
            hd95_list.append(hd95)
            sen_list.append(sen)

            large_inter += np.logical_and(large_pred.detach().cpu().numpy() > 0.,
                                          large_trace.detach().cpu().numpy() > 0.).sum()
            large_union += np.logical_or(large_pred.detach().cpu().numpy() > 0.,
                                         large_trace.detach().cpu().numpy() > 0.).sum()
            small_inter += np.logical_and(small_pred.detach().cpu().numpy() > 0.,
                                          small_trace.detach().cpu().numpy() > 0.).sum()
            small_union += np.logical_or(small_pred.detach().cpu().numpy() > 0.,
                                         small_trace.detach().cpu().numpy() > 0.).sum()

            large_inter_list.extend(
                np.logical_and(large_pred.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum(
                    (1, 2)))
            large_union_list.extend(
                np.logical_or(large_pred.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum(
                    (1, 2)))
            small_inter_list.extend(
                np.logical_and(small_pred.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum(
                    (1, 2)))
            small_union_list.extend(
                np.logical_or(small_pred.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum(
                    (1, 2)))

        return assd_list, hd95_list, sen_list, np.array(large_inter_list), np.array(large_union_list), np.array(
            small_inter_list), np.array(small_union_list)


    def compute_asd(self, true_mask, pred_mask):
        true_mask = true_mask.squeeze().cpu().numpy()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        true_labels = measure.label(true_mask)
        pred_labels = measure.label(pred_mask)
        true_contours = measure.find_contours(true_mask, 0.5)
        pred_contours = measure.find_contours(pred_mask, 0.5)
        distances = []
        for pred_contour in pred_contours:
            for point in pred_contour:
                x, y = point[0], point[1]
                nearest_dist = self.nearest_distance(x, y, true_contours)
                distances.append(nearest_dist)
        if not distances: return 0.0
        return np.mean(distances)

    def nearest_distance(self, x, y, contours):
        min_dist = float('inf')
        for contour in contours:
            for point in contour:
                dist = np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
                min_dist = min(min_dist, dist)
        return min_dist

    def compute_sen(self, true_mask, pred_mask):
        true_mask = true_mask.squeeze().cpu().numpy()
        pred_mask = pred_mask.squeeze().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(true_mask.flatten(), pred_mask.flatten()).ravel()
        if tp + fn == 0:
            return 1.0
        return tp / (tp + fn)

    def compute_assd(self, true_mask, pred_mask):
        pred = (pred_mask > 0.5).cpu().numpy()
        target = (true_mask > 0.5).cpu().numpy()
        try:
            return medpy_assd(pred.squeeze(), target.squeeze(), voxelspacing=[1.0, 1.0])
        except:
            return 0.0

    def compute_hd95(self, true_mask, pred_mask):
        pred = (pred_mask > 0.5).cpu().numpy()
        target = (true_mask > 0.5).cpu().numpy()
        try:
            return medpy_hd95(pred.squeeze(), target.squeeze(), voxelspacing=[1.0, 1.0])
        except:
            return 0.0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="处理数据、预测结果和输出目录的路径")
    parser.add_argument('--data_dir', type=str, required=True, default='/workdir1/echo_dataset/EchoNet-Dynamic', help='EchoNet Dynamic Dataset的路径')
    parser.add_argument('--prediction_dir', type=str, required=True, help='预测结果存放的目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='结果保存的输出目录路径')

    args = parser.parse_args()

    test_model = TestData()
    test_model.run_test(data_dir=args.data_dir, pred_dir=args.prediction_dir, output_dir=args.output_dir)
