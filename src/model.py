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
    """Dual-encoder LSTM with temporal self-attention over aux hidden states.

    Architecture:
      - Main encoder: LSTM (12-month sea ice) → main_hidden
      - Aux encoder:  LSTM (last N months of lagged climate indices)
                      → self-attention pooling over ALL time steps
                      → aux_hidden (weighted sum)
      - The self-attention mechanism lets the model learn, for example,
        that "NAO at lag-3" matters more than "NAO at lag-1" for sea ice
        prediction, without requiring manual lag selection.
      - Concat(main_hidden, aux_hidden) → Linear → output_len
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

        # Aux encoder: processes last N months of lagged climate indices
        self.aux_lstm = nn.LSTM(
            input_size=aux_input_size,
            hidden_size=aux_hidden,
            num_layers=1,  # shallow — aux data is short and noisy
            batch_first=True,
        )
        self.aux_dropout = nn.Dropout(aux_dropout)  # stronger dropout on aux

        # Temporal self-attention over aux hidden states (Bahdanau-style):
        # A two-layer MLP with tanh activation learns a non-linear scalar
        # importance score for each time step of the aux LSTM output, then
        # softmax-normalizes across time. The non-linearity lets the model
        # express "lag-3 is far more important than lag-1" — a single linear
        # layer could only express a monotone re-weighting of the hidden
        # dimensions and produced near-uniform weights in practice.
        self.aux_attn = nn.Sequential(
            nn.Linear(aux_hidden, aux_hidden // 2),
            nn.Tanh(),
            nn.Linear(aux_hidden // 2, 1),
        )

        self.aux_seq_len = aux_seq_len

        # Fusion layer
        self.fc = nn.Linear(main_hidden + aux_hidden, output_len)

        # Store latest attention weights for interpretability (not used in forward)
        self._last_attn_weights = None

        self._init_weights()

    def _init_weights(self):
        for lstm in [self.main_lstm, self.aux_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        for layer in self.aux_attn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_main, x_aux):
        """
        Args:
            x_main: (batch, 12, 1) — sea ice area sequence
            x_aux:  (batch, aux_seq_len, aux_input_size) — lagged climate indices
        Returns:
            (batch, output_len)
        """
        # Main encoder
        main_out, _ = self.main_lstm(x_main)
        main_h = main_out[:, -1, :]  # (batch, main_hidden)
        main_h = self.main_dropout(main_h)

        # Aux encoder → all hidden states
        aux_out, _ = self.aux_lstm(x_aux)  # (batch, aux_seq_len, aux_hidden)

        # Temporal self-attention pooling
        # attn_scores: (batch, aux_seq_len, 1) — importance of each time step
        attn_scores = self.aux_attn(aux_out)
        attn_weights = F.softmax(attn_scores, dim=1)  # normalize over time
        aux_h = torch.sum(attn_weights * aux_out, dim=1)  # (batch, aux_hidden)

        # Store for interpretability (detach to avoid graph retention)
        self._last_attn_weights = attn_weights.detach().squeeze(-1)  # (batch, aux_seq_len)

        aux_h = self.aux_dropout(aux_h)

        # Fusion
        combined = torch.cat([main_h, aux_h], dim=1)
        out = self.fc(combined)
        return out

    def get_attention_weights(self):
        """Return attention weights from the last forward pass.

        Returns:
            (batch, aux_seq_len) tensor, or None if no forward pass has been run.
            Each row sums to 1.0 — higher values indicate time steps the model
            relied on more heavily for the aux representation.
        """
        return self._last_attn_weights


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