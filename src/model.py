# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    """基础线性回归模型"""

    def __init__(self, input_len=12, output_len=12):
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(input_len, output_len)

    def forward(self, x):
        x = x.squeeze(-1)
        out = self.fc(x)
        return out


class SimpleRNNModel(nn.Module):
    """基础SimpleRNN模型"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=1,
                 output_len=12, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out


class SeaIceLSTM(nn.Module):
    """优化的LSTM模型（支持简化模式）"""

    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_len=12, dropout=0.1):
        super(SeaIceLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 简化模式：当num_layers=1时使用极简结构
        if num_layers == 1:
            # 极简单层LSTM（类似SimpleRNN结构）
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_len)
            self.is_simple = True

        else:
            # 原复杂结构（多层LSTM+特征提取）
            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )

            self.lstm2 = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=dropout
            )

            # 层归一化（提高训练稳定性）
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)

            # 季节性特征提取分支
            self.seasonal_fc = nn.Linear(hidden_size, hidden_size // 2)
            self.trend_fc = nn.Linear(hidden_size, hidden_size // 2)

            # 输出层
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_len)
            self.relu = nn.ReLU()
            self.is_simple = False

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        if self.is_simple:
            # 简化模式：初始化单层LSTM和输出层
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

            # 输出层初始化
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        else:
            # 复杂模式：初始化原结构
            # 初始化第一层LSTM
            for name, param in self.lstm1.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

            # 初始化第二层LSTM
            for name, param in self.lstm2.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

            # 全连接层初始化
            nn.init.xavier_uniform_(self.seasonal_fc.weight)
            nn.init.zeros_(self.seasonal_fc.bias)

            nn.init.xavier_uniform_(self.trend_fc.weight)
            nn.init.zeros_(self.trend_fc.bias)

            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        if self.is_simple:
            # 简化模式：单层LSTM + Dropout + 线性层
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
            last_out = self.dropout(last_out)
            out = self.fc(last_out)
            return out

        else:
            # 复杂模式：原多层LSTM+特征提取结构
            # 第一层LSTM
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.layer_norm1(lstm_out)

            # 第二层LSTM
            lstm_out, _ = self.lstm2(lstm_out)
            lstm_out = self.layer_norm2(lstm_out)

            # 取最后一个时间步的输出
            last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

            # 分离季节性和趋势特征
            seasonal_features = self.relu(self.seasonal_fc(last_out))
            trend_features = self.relu(self.trend_fc(last_out))

            # 合并特征
            combined = torch.cat([seasonal_features, trend_features], dim=1)
            combined = self.dropout(combined)

            # 输出层
            out = self.fc(combined)  # (batch, output_len)

            return out


class SeaIceDualEncoderLSTM(nn.Module):
    """Dual-encoder LSTM: main encoder for sea ice, aux encoder for lagged AO/SST.

    Architecture:
      - Main encoder: LSTM (12-month sea ice)  → main_hidden
      - Aux encoder:  LSTM (last N months of lagged AO/SST) → aux_hidden
      - Extra dropout on aux path to suppress noise
      - Concat(main_hidden, aux_hidden) → Linear → output_len

    This prevents AO/SST noise from contaminating the main LSTM's recurrent
    state across all 12 time steps. The aux encoder only sees the most recent
    months, capturing short-term atmospheric/oceanic influence on sea ice.
    """

    def __init__(self, input_size=1, main_hidden=256, aux_hidden=64,
                 aux_input_size=7, aux_seq_len=3, num_layers=1,
                 output_len=12, dropout=0.1, aux_dropout=0.3):
        super(SeaIceDualEncoderLSTM, self).__init__()

        # Main encoder: processes full 12-month sea ice sequence
        self.main_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=main_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.main_dropout = nn.Dropout(dropout)

        # Aux encoder: processes last N months of lagged AO/SST
        self.aux_lstm = nn.LSTM(
            input_size=aux_input_size,
            hidden_size=aux_hidden,
            num_layers=1,  # shallow — aux data is short and noisy
            batch_first=True,
        )
        self.aux_dropout = nn.Dropout(aux_dropout)  # stronger dropout on aux

        self.aux_seq_len = aux_seq_len

        # Fusion layer
        self.fc = nn.Linear(main_hidden + aux_hidden, output_len)

        self._init_weights()

    def _init_weights(self):
        for lstm in [self.main_lstm, self.aux_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_main, x_aux):
        """
        Args:
            x_main: (batch, 12, 1) — sea ice area sequence
            x_aux:  (batch, aux_seq_len, aux_input_size) — lagged AO/SST
        Returns:
            (batch, output_len)
        """
        # Main encoder
        main_out, _ = self.main_lstm(x_main)
        main_h = main_out[:, -1, :]  # (batch, main_hidden)
        main_h = self.main_dropout(main_h)

        # Aux encoder
        aux_out, _ = self.aux_lstm(x_aux)
        aux_h = aux_out[:, -1, :]  # (batch, aux_hidden)
        aux_h = self.aux_dropout(aux_h)

        # Fusion
        combined = torch.cat([main_h, aux_h], dim=1)
        out = self.fc(combined)
        return out


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码
if __name__ == "__main__":
    model = SeaIceLSTM(input_size=1, hidden_size=64, output_len=12)
    print(model)
    print(f"参数量: {count_parameters(model):,}")

    # 测试前向传播
    test_input = torch.randn(32, 12, 1)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")