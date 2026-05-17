import pandas as pd
import os

class VolumeTracingsCleaner:
    def __init__(self, input_file_path, output_file_path=None):
        """
        初始化 VolumeTracingsCleaner 类。

        参数:
        input_file_path (str): 输入的 VolumeTracings.csv 文件路径。
        output_file_path (str, optional): 输出的清理后的 CSV 文件路径。如果为 None，则默认为 'input_file_path。
        """
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path if output_file_path else input_file_path
        self.backup_file_path = os.path.dirname(input_file_path)+'/VolumeTracings_origin.csv'

    def clean_data(self):
        """
        清理 VolumeTracings.csv 文件中 X 或 Y 列值为空或者 Frame 列为 'No Systolic' 的行。
        """
        # 读取 CSV 文件
        df = pd.read_csv(self.input_file_path)

        # 备份原文件
        self._backup_original_file()

        # 删除 X 或 Y 列值为空的行
        df_cleaned = df.dropna(subset=['X', 'Y'])

        # 删除 Frame 列为 'No Systolic' 的行
        df_cleaned = df_cleaned[df_cleaned['Frame'] != 'No Systolic']

        # 因为读取的视频数据从0帧开始编号，因此使 Frame 列中的所有数据减一
        df_cleaned['Frame'] = df_cleaned['Frame'].astype(int) - 1

        # 保存清理后的数据到新的 CSV 文件
        df_cleaned.to_csv(self.output_file_path, index=False)

    def _backup_original_file(self):
        """
        备份原文件。
        """
        if not os.path.exists(self.backup_file_path):
            os.rename(self.input_file_path, self.backup_file_path)
        else:
            raise FileExistsError(f"备份文件 {self.backup_file_path} 已存在，请手动处理。")

# 示例用法
if __name__ == "__main__":
    cleaner = VolumeTracingsCleaner('/workdir1/cn24/data/pediatric_echo/A4C/VolumeTracings.csv')
    cleaner.clean_data()
