"""
Dedicated Optuna tuning for TRIVARIATE LSTM (sea ice + AO + SST, input_size=3).

Tunes for all 3 output lengths (short/medium/long).
Saves best params to outputs/results/trivariate_best_params.json
— does NOT modify config.py.
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
    load_trivariate_data, create_sequences, train_val_test_split
)
from src.dataset import SeaIceDataset
from src.model import SeaIceLSTM
from src.utils import set_seed


AO_CSV = os.path.join(config.BASE_DIR, "data", "ao_monthly.csv")
SST_CSV = os.path.join(config.BASE_DIR, "data", "arctic_sst_monthly.csv")

TUNING_CONFIGS = [
    {"output_len": 1,  "study_name": "trivariate_v2_out1",  "db": "optuna_trivariate_v2_out1.db",  "n_trials": 40},
    {"output_len": 6,  "study_name": "trivariate_v2_out6",  "db": "optuna_trivariate_v2_out6.db",  "n_trials": 40},
    {"output_len": 12, "study_name": "trivariate_v2_out12", "db": "optuna_trivariate_v2_out12.db", "n_trials": 40},
]


def make_objective(output_len):
    """Factory: returns Optuna objective for trivariate model at given output_len."""

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        set_seed(42)

        df, sc_ice, sc_ao, sc_sst = load_trivariate_data(
            config.DATA_DIR, AO_CSV, SST_CSV, config.TARGET_COLUMN
        )

        data_input = df[[f'{config.TARGET_COLUMN}_scaled', 'ao_scaled', 'sst_scaled']].values
        data_target = df[f'{config.TARGET_COLUMN}_scaled'].values
        X, y = create_sequences(data_input, config.INPUT_LEN, output_len,
                                target_data=data_target)

        (X_train, y_train), (X_val, y_val), _ = train_val_test_split(X, y, 0.7, 0.15)

        train_ds = SeaIceDataset(X_train, y_train)
        val_ds = SeaIceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SeaIceLSTM(
            input_size=3,
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

        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_loss += criterion(model(batch_X), batch_y).item()
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


def run_all():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULTS_DIR, "trivariate_best_params.json")

    # Remove old DBs to force fresh tuning
    for cfg in TUNING_CONFIGS:
        db_path = os.path.join(config.BASE_DIR, cfg["db"])
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed old DB: {db_path}")

    all_best = {}

    for cfg in TUNING_CONFIGS:
        ol = cfg["output_len"]
        print(f"\n{'='*60}")
        print(f"  Trivariate Optuna Tuning: output_len = {ol}")
        print(f"  Trials: {cfg['n_trials']}")
        print(f"{'='*60}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # less noise

        study = optuna.create_study(
            direction='minimize',
            study_name=cfg["study_name"],
            storage=f'sqlite:///{cfg["db"]}',
        )

        study.optimize(make_objective(ol), n_trials=cfg["n_trials"],
                       show_progress_bar=True)

        best = study.best_params
        best["best_val_loss"] = study.best_value
        all_best[str(ol)] = best

        print(f"\n  Best (out={ol}): val_loss={study.best_value:.6f}")
        for k, v in best.items():
            if k != "best_val_loss":
                print(f"    {k}: {v}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("  Trivariate Best Params Summary")
    print(f"{'='*70}")
    param_keys = ['hidden_size', 'num_layers', 'dropout', 'learning_rate',
                  'batch_size', 'weight_decay']
    header = f"{'Param':<20}"
    for ol in sorted(all_best.keys(), key=int):
        header += f"{'out=' + ol:<24}"
    print(header)
    print("-" * (20 + 24 * len(all_best)))
    for pk in param_keys:
        row = f"{pk:<20}"
        for ol in sorted(all_best.keys(), key=int):
            row += f"{str(all_best[ol][pk]):<24}"
        print(row)


if __name__ == "__main__":
    run_all()
