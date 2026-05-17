import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class ImageVisualizer:
    def __init__(self):
        pass

    def show_image(self, image, title=None, cmap=None, colorbar=False, show_axis=False, save_path=''):
        """
        可视化单张图像，支持 PIL 图像、Numpy 数组和 PyTorch 张量。
        :param image: 输入图像，可以是 PIL 图像、Numpy 数组或 PyTorch 张量
        :param title: 图像标题
        :param cmap: 颜色映射，用于灰度图像或其他模式图像（默认为None）
        :param colorbar: 是否显示颜色条，适用于灰度图像（默认为False）
        :param show_axis: 是否显示坐标轴（默认为False）
        """
        image = self._prepare_image(image)

        plt.imshow(image, cmap=cmap)

        if title:
            plt.title(title)

        if colorbar:
            plt.colorbar()

        if not show_axis:
            plt.axis('off')  # 隐藏多余的子图
        if save_path:
            # 保存图像到指定路径
            plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' 去掉多余的边框
            print(f"图像已保存到 {save_path}")


        plt.show()

    def show_images(self, images, titles=None, cmap=None, colorbar=False, show_axis=False, max_cols=4, save_path=None):
        """
        可视化多张图像，支持 PIL 图像、Numpy 数组和 PyTorch 张量。
        :param images: 输入图像列表，可以是 PIL 图像、Numpy 数组或 PyTorch 张量的列表
        :param titles: 图像标题列表
        :param cmap: 颜色映射，用于灰度图像或其他模式图像（默认为None）
        :param colorbar: 是否显示颜色条，适用于灰度图像（默认为False）
        :param show_axis: 是否显示坐标轴（默认为False）
        :param max_cols: 每行最多显示多少张图像
        :param save_path:  保存路径
        """
        num_images = len(images)
        max_cols = max_cols  # 每行最多显示max_cols张图像
        num_rows = (num_images + max_cols - 1) // max_cols  # 计算总行数

        fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * max_cols, max_cols * num_rows))

        # Flatten axes in case of a multi-row layout, for easy iteration
        axes = axes.flatten() if num_images > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < num_images:
                # Prepare and display each image
                image = self._prepare_image(images[i])
                ax.imshow(image, cmap=cmap)

                if titles and len(titles) > i:
                    ax.set_title(titles[i])

                if colorbar:
                    plt.colorbar(ax.imshow(image, cmap=cmap), ax=ax)

                if not show_axis:
                    ax.set_xticks([])
                    ax.set_yticks([])

            else:
                ax.axis('off')  # 隐藏多余的子图
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图之间的间隔
        if save_path:
            # 保存图像到指定路径
            plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' 去掉多余的边框
            print(f"图像已保存到 {save_path}")
        plt.show()

    def save_image(self, image, file_path):
        """
        保存图像到文件
        :param image: 输入图像，可以是 PIL 图像、Numpy 数组或 PyTorch 张量
        :param file_path: 保存图像的路径
        """
        image = self._prepare_image(image)
        pil_image = Image.fromarray(np.uint8(image * 255))  # 转换为 8-bit 格式
        pil_image.save(file_path)

    def _prepare_image(self, image):
        """
        将输入的图像转换为可视化的格式 (H, W, C) 或 (H, W)。
        :param image: 输入图像
        :return: 处理后的 Numpy 数组格式的图像
        """
        if isinstance(image, torch.Tensor):
            # 如果是 PyTorch 张量，转换为 Numpy 格式并去掉 batch 维度
            image = image.squeeze().detach().cpu().numpy()
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        elif isinstance(image, Image.Image):
            # 如果是 PIL 图像，转换为 Numpy 数组
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass  # 如果已经是 Numpy 数组，直接使用
        else:
            raise TypeError("Unsupported image type. Expected PIL, Numpy array, or PyTorch Tensor.")

        # 如果是灰度图像，确保维度为 (H, W) 而不是 (H, W, 1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)

        return image
