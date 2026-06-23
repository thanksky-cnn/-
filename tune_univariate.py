# optuna_tuning.py
"""
使用 Optuna 自动调参
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import config
from src.data_preprocessing import load_and_merge_data, create_sequences, train_val_test_split
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM
from src.train import train_model, predict
from src.utils import set_seed


def objective(trial):
    """
    Optuna 的目标函数
    trial 会尝试不同的参数组合
    """
    # 建议超参数
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    print(f"\n尝试参数组合: hidden_size={hidden_size}, dropout={dropout}, lr={learning_rate:.6f}")

    # 固定随机种子
    set_seed(42)

    # 1. 加载数据
    df, scaler = load_and_merge_data(config.DATA_DIR, config.TARGET_COLUMN)
    data_scaled = df[f'{config.TARGET_COLUMN}_scaled'].values
    X, y = create_sequences(data_scaled, config.INPUT_LEN, config.OUTPUT_LEN)

    # 2. 划分数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, 0.7, 0.15
    )

    # 3. 创建 DataLoader
    train_dataset = SeaIceDataset(X_train, y_train)
    val_dataset = SeaIceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 4. 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SeaIceLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_len=config.OUTPUT_LEN,
        dropout=dropout
    ).to(device)

    # 5. 训练
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 简化训练（只训练50个epoch用于快速调参）
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):  # 快速调参，减少epoch
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

    print(f"  验证损失: {best_val_loss:.6f}")

    # 返回验证损失（Optuna会最小化这个值）
    return best_val_loss


def run_optuna_tuning():
    """运行 Optuna 调参"""

    print("=" * 60)
    print("Optuna 自动调参")
    print("=" * 60)
    print("\nOptuna 会尝试不同的参数组合，找到最佳配置...")
    print("这可能需要一些时间，请耐心等待...\n")

    # 创建研究
    study = optuna.create_study(
        direction='minimize',  # 最小化验证损失
        study_name='seaice_lstm_tuning',
        storage='sqlite:///optuna_study.db',  # 保存结果
        load_if_exists=True
    )

    # 运行优化
    study.optimize(objective, n_trials=30)  # 尝试30组参数

    # 输出结果
    print("\n" + "=" * 60)
    print("调参完成！最佳参数：")
    print("=" * 60)

    best_params = study.best_params
    best_value = study.best_value

    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  最佳验证损失: {best_value:.6f}")

    # 显示所有试验结果
    print("\n" + "=" * 60)
    print("所有试验结果（按损失排序）：")
    print("=" * 60)

    df_results = study.trials_dataframe()
    df_results = df_results.sort_values('value')
    print(df_results[['number', 'value'] + [f'params_{k}' for k in best_params.keys()]].head(10))

    # 保存结果
    df_results.to_csv('optuna_results.csv', index=False)
    print("\n结果已保存到 optuna_results.csv")

    return study


if __name__ == "__main__":
    study = run_optuna_tuning()