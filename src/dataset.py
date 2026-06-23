# src/dataset.py
import torch
from torch.utils.data import Dataset


class SeaIceDataset(Dataset):
    """海冰数据PyTorch Dataset"""

    def __init__(self, X, y):
        """
        Args:
            X: 输入特征数组 (n_samples, input_len, 1)
            y: 目标数组 (n_samples, output_len)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 测试代码
if __name__ == "__main__":
    import numpy as np

    # 模拟数据
    X_test = np.random.randn(100, 12, 1)
    y_test = np.random.randn(100, 12)
    dataset = SeaIceDataset(X_test, y_test)
    print(f"数据集大小: {len(dataset)}")
    x_sample, y_sample = dataset[0]
    print(f"样本X形状: {x_sample.shape}")
    print(f"样本y形状: {y_sample.shape}")