
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class WrongPredictionsDataset(Dataset):
    """
    自定义 PyTorch Dataset，用于加载 wrong_predictions 数据。
    每个样本是一个元组 (image, true_label, pred_label)。
    """
    def __init__(self, wrong_predictions, transform=None):
        """
        初始化 Dataset。
        :param wrong_predictions: 加载的 wrong_predictions 数据（列表，每个元素是元组）。
        :param transform: 可选的图像变换（例如数据增强）。
        """
        self.wrong_predictions = wrong_predictions
        self.transform = transform

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.wrong_predictions)

    def __getitem__(self, idx):
        """
        根据索引返回单个样本。
        :param idx: 样本索引。
        :return: 图像、真实标签和预测标签。
        """
        image, true_label, pred_label = self.wrong_predictions[idx]

        # 如果定义了 transform，对图像进行变换
        if self.transform:
            image = self.transform(image)

        return image, true_label, pred_label
    
# 读取数据的小demo
file_path = "water2/out/TSCeption_hidden-bit=30/wrong_predictions_130_72.pkl" # 假设数据存储在该文件中
with open(file_path, 'rb') as f:
    wrong_predictions = pickle.load(f)
for i, (image, true_label, pred_label) in enumerate(wrong_predictions):  # 打印前 5 个样本
    print(f"\nSample {i + 1}:")
    print("Image shape:", image)  # 打印图像形状
    print("True label:", true_label)  # 打印真实标签
    print("Predicted label:", pred_label)  # 打印预测标签