# src/hybrid_model.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TinyLSTM(nn.Module):
    """极简LSTM，专门学习残差"""

    def __init__(self, input_size=1, hidden_size=16, output_len=12):
        super(TinyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_len = output_len

        # 极简LSTM结构
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_len)
        self.dropout = nn.Dropout(0.2)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias'