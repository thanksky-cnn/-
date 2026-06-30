# src/train.py
import torch
import time
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip_norm=None):
    """训练一个epoch（添加梯度裁剪）"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, config, device):
    """
    完整训练流程（添加学习率调度和正则化）
    """
    criterion = nn.MSELoss()

    # 添加L2正则化（weight_decay）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 学习率调度器（当验证损失不再下降时降低学习率）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,

    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{config.MODELS_DIR}/best_model.pth"
    best_epoch = None
    train_start = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 device, config.GRAD_CLIP_NORM)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch < 5:  # 更频繁地打印训练进度
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.6f}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at Epoch {epoch + 1}")
            break

    train_time_s = time.time() - train_start
    return train_losses, val_losses, best_model_path, best_epoch, train_time_s


def predict(model, X, device, scaler=None):
    """预测并可选反归一化"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    if scaler:
        y_pred = scaler.inverse_transform(y_pred_scaled)
        return y_pred
    return y_pred_scaled


# ============================================================
# 双编码器训练函数（供 Phase 1/2/3 实验使用）
# ============================================================

def train_dual_encoder(model, train_loader, val_loader, hp, device,
                        num_epochs=1000, early_stopping_patience=30,
                        verbose=True):
    """
    训练双编码器模型（支持自定义超参数 dict）。

    Args:
        model: SeaIceDualEncoderLSTM 实例
        train_loader: DEDataset 的 DataLoader
        val_loader: DEDataset 的 DataLoader
        hp: 超参数字典，需包含 learning_rate, weight_decay
        device: torch device
        num_epochs: 最大训练轮数
        early_stopping_patience: 早停耐心值
        verbose: 是否打印训练进度

    Returns:
        train_losses, val_losses, best_epoch, train_time_s
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.get("learning_rate", 0.000635),
        weight_decay=hp.get("weight_decay", 2.59e-06)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=15
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for Xm_b, Xa_b, yb in train_loader:
            Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xm_b, Xa_b), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xm_b, Xa_b, yb in val_loader:
                Xm_b, Xa_b, yb = Xm_b.to(device), Xa_b.to(device), yb.to(device)
                val_loss += criterion(model(Xm_b, Xa_b), yb).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % 20 == 0 or epoch < 3):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.6f}")

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time_s = time.time() - train_start

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return train_losses, val_losses, best_state, best_epoch, train_time_s


def predict_dual_encoder(model, X_main, X_aux, device, scaler=None):
    """
    双编码器模型预测。

    Args:
        model: SeaIceDualEncoderLSTM 实例
        X_main: (N, 12, 1) 冰序列
        X_aux: (N, aux_seq_len, N_channels) 辅助特征
        device: torch device
        scaler: 可选，MinMaxScaler 用于反归一化

    Returns:
        y_pred: (N, output_len) 预测值（反归一化后）
    """
    model.eval()
    Xm_t = torch.tensor(X_main, dtype=torch.float32).to(device)
    Xa_t = torch.tensor(X_aux, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_scaled = model(Xm_t, Xa_t).cpu().numpy()

    if scaler:
        # scaler expects (N, 1) shape, but y_pred_scaled is (N, output_len)
        # Reshape, inverse transform, reshape back
        n, out_len = y_pred_scaled.shape
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred.reshape(n, out_len)
    return y_pred_scaled
