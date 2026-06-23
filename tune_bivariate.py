"""
Optuna hyperparameter tuning for BIVARIATE LSTM (sea ice + AO, input_size=2).

Tunes for all 3 output lengths (short/medium/long) separately.
Saves best params to outputs/results/bivariate_best_params.json
— does NOT modify config.py, preserving original univariate params.

Uses ratio-based split (70/15/15) for speed, same as optuna_tuning.py.
"""
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import optuna
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import config
from src.data_preprocessing import (
    load_multivariate_data, create_sequences, train_val_test_split
)
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM
from src.utils import set_seed


AO_CSV = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")

# Tuning configs mirroring _SCHEME_PARAMS output lengths
TUNING_CONFIGS = [
    {"output_len": 1,  "study_name": "bivariate_lstm_out1",  "db": "optuna_bivariate_out1.db", "n_trials": 30},
    {"output_len": 6,  "study_name": "bivariate_lstm_out6",  "db": "optuna_bivariate_out6.db", "n_trials": 30},
    {"output_len": 12, "study_name": "bivariate_lstm_out12", "db": "optuna_bivariate_out12.db", "n_trials": 30},
]


def make_objective(output_len):
    """Factory: returns an Optuna objective function for the given output_len."""

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        set_seed(42)

        # Load multivariate data
        df, scaler_ice, scaler_ao = load_multivariate_data(
            config.DATA_DIR, AO_CSV, config.TARGET_COLUMN
        )

        # Bivariate sequences: X has [area, ao], y is area only
        data_input = df[[f'{config.TARGET_COLUMN}_scaled', 'ao_scaled']].values
        data_target = df[f'{config.TARGET_COLUMN}_scaled'].values
        X, y = create_sequences(data_input, config.INPUT_LEN, output_len,
                                target_data=data_target)

        (X_train, y_train), (X_val, y_val), _ = train_val_test_split(X, y, 0.7, 0.15)

        train_dataset = SeaIceDataset(X_train, y_train)
        val_dataset = SeaIceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SeaIceLSTM(
            input_size=2,  # sea ice + AO
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_len=output_len,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(50):  # fast tuning
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 10:
                break

        return best_val_loss

    return objective


def run_all_tunings():
    """Run Optuna tuning for all 3 output lengths, save best params to JSON."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULTS_DIR, "bivariate_best_params.json")

    all_best = {}

    for cfg in TUNING_CONFIGS:
        ol = cfg["output_len"]
        print(f"\n{'='*60}")
        print(f"  Bivariate Tuning: output_len = {ol}")
        print(f"  Study: {cfg['study_name']}, Trials: {cfg['n_trials']}")
        print(f"{'='*60}")

        study = optuna.create_study(
            direction='minimize',
            study_name=cfg["study_name"],
            storage=f'sqlite:///{cfg["db"]}',
            load_if_exists=True,
        )

        study.optimize(make_objective(ol), n_trials=cfg["n_trials"])

        best = study.best_params
        best["best_val_loss"] = study.best_value
        all_best[str(ol)] = best

        print(f"  Best (out={ol}): val_loss={study.best_value:.6f}")
        for k, v in best.items():
            if k != "best_val_loss":
                print(f"    {k}: {v}")

    # Save to JSON (not touching config.py)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)
    print(f"\nBest bivariate params saved to: {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("  Summary: Best Params for Bivariate LSTM")
    print(f"{'='*70}")
    param_keys = ['hidden_size', 'num_layers', 'dropout', 'learning_rate',
                  'batch_size', 'weight_decay']
    header = f"{'Param':<20}"
    for ol in sorted(all_best.keys(), key=int):
        header += f"{'out=' + ol:<22}"
    print(header)
    print("-" * (20 + 22 * len(all_best)))
    for pk in param_keys:
        row = f"{pk:<20}"
        for ol in sorted(all_best.keys(), key=int):
            row += f"{str(all_best[ol][pk]):<22}"
        print(row)

    return all_best


if __name__ == "__main__":
    run_all_tunings()
