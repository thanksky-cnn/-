"""Shared experiment result data for all thesis figure generation scripts.

Single source of truth — all gen_fig_*.py scripts import from here.
Data extracted from outputs/results/phase{1,2,3,4}/*.json (2026-07 verified).
"""

# ===========================================================================
# Phase 1: Single-Variable Increment (unified E7v1 hyperparams, 5-seed ensemble)
# ===========================================================================
PHASE1_SINGLE_VAR = {
    "E1":  {"label": "纯冰 LSTM (E1)",    "aux": "无",                "rmse": 0.5236, "rmse_mean": 0.5319, "rmse_std": 0.0115, "delta":  0.0000, "category": "baseline"},
    "E10": {"label": "+PNA (E10)",         "aux": "PNA (太平洋)","rmse": 0.5089, "rmse_mean": 0.5252, "rmse_std": 0.0149, "delta": -0.0147, "category": "single"},
    "E14": {"label": "+AO (E14)",          "aux": "AO (北极)","rmse": 0.5130, "rmse_mean": 0.5280, "rmse_std": 0.0155, "delta": -0.0106, "category": "single"},
    "E9":  {"label": "+Nino3.4 (E9)",      "aux": "Nino3.4 (热带)","rmse": 0.5180, "rmse_mean": 0.5256, "rmse_std": 0.0154, "delta": -0.0055, "category": "single"},
    "E8":  {"label": "+NAO (E8)",          "aux": "NAO (大西洋)","rmse": 0.5192, "rmse_mean": 0.5315, "rmse_std": 0.0147, "delta": -0.0044, "category": "single"},
    "E7v1":{"label": "+AO+SST (E7v1)","aux": "AO+SST","rmse": 0.5420, "rmse_mean": 0.5512, "rmse_std": 0.0215, "delta": +0.0184, "category": "degradation"},
}

# ===========================================================================
# Phase 2: Variable Combinations
# ===========================================================================
PHASE2_COMBINATIONS = {
    "E17": {"label": "SST+Nino3.4 (E17)","aux_vars": ["sst","nino34"], "n_vars": 2, "type": "海洋内部",  "rmse": 0.5098, "rmse_mean": 0.5228, "rmse_std": 0.0101, "delta": -0.0138},
    "E18": {"label": "AO+SST+NAO (E18)",  "aux_vars": ["ao","sst","nao"], "n_vars": 3, "type": "大气+海洋","rmse": 0.5218, "rmse_mean": 0.5354, "rmse_std": 0.0145, "delta": -0.0018},
    "E15": {"label": "AO+NAO (E15)",      "aux_vars": ["ao","nao"], "n_vars": 2, "type": "同扇区","rmse": 0.5229, "rmse_mean": 0.5287, "rmse_std": 0.0074, "delta": -0.0007},
    "E19": {"label": "全五变量 (E19)","aux_vars": ["ao","sst","nao","pna","nino34"], "n_vars": 5, "type": "全部","rmse": 0.5243, "rmse_mean": 0.5372, "rmse_std": 0.0191, "delta": +0.0007},
    "E23": {"label": "AO+PNA (E23)",      "aux_vars": ["ao","pna"], "n_vars": 2, "type": "跨扇区","rmse": 0.5244, "rmse_mean": 0.5366, "rmse_std": 0.0221, "delta": +0.0008},
    "E16": {"label": "AO+Nino3.4 (E16)","aux_vars": ["ao","nino34"], "n_vars": 2, "type": "极地+热带","rmse": 0.5272, "rmse_mean": 0.5378, "rmse_std": 0.0175, "delta": +0.0036},
    "E24": {"label": "AO+NAO+PNA (E24)","aux_vars": ["ao","nao","pna"], "n_vars": 3, "type": "全大气","rmse": 0.5383, "rmse_mean": 0.5528, "rmse_std": 0.0170, "delta": +0.0147},
    "E34": {"label": "PNA+NAO (E34)",    "aux_vars": ["pna","nao"], "n_vars": 2, "type": "跨太平洋-大西洋","rmse": 0.5370, "rmse_mean": 0.5488, "rmse_std": 0.0197, "delta": +0.0134},
}

# ===========================================================================
# Phase 3: Ablation (baseline = E19 ALL5 = 0.5243)
# ===========================================================================
PHASE3_ABLATION = {
    "E25": {"label": "移除 PNA (E25)","removed": "PNA",   "rmse": 0.5154, "rmse_mean": 0.5313, "rmse_std": 0.0102, "delta": -0.0089, "direction": "improve"},
    "E20": {"label": "移除 NAO (E20)","removed": "NAO",   "rmse": 0.5161, "rmse_mean": 0.5308, "rmse_std": 0.0114, "delta": -0.0081, "direction": "improve"},
    "E22": {"label": "移除 SST (E22)","removed": "SST",   "rmse": 0.5209, "rmse_mean": 0.5329, "rmse_std": 0.0086, "delta": -0.0034, "direction": "improve"},
    "E21": {"label": "移除 Nino3.4 (E21)","removed": "Nino3.4","rmse": 0.5337, "rmse_mean": 0.5489, "rmse_std": 0.0111, "delta": +0.0095, "direction": "degrade"},
    "E33": {"label": "移除 AO (E33)",   "removed": "AO",     "rmse": 0.5138, "rmse_mean": 0.5297, "rmse_std": 0.0053, "delta": -0.0104, "direction": "improve"},
}
PHASE3_BASELINE_RMSE = 0.5243  # E19 ALL5

# ===========================================================================
# Phase 4: Regional Spatial Heterogeneity
# ===========================================================================

# Region metadata
REGIONS = [
    {"id": "bering",    "name": "白令海",    "sector": "Pacific",  "sector_cn": "太平洋扇区", "match_idx": "PNA", "mean_area": 0.30},
    {"id": "chukchi",   "name": "楚科奇海","sector": "Pacific",  "sector_cn": "太平洋扇区", "match_idx": "PNA", "mean_area": 0.42},
    {"id": "barents",   "name": "巴伦支海","sector": "Atlantic","sector_cn": "大西洋扇区", "match_idx": "NAO", "mean_area": 0.47},
    {"id": "kara",      "name": "喀拉海",    "sector": "Atlantic","sector_cn": "大西洋扇区", "match_idx": "NAO", "mean_area": 0.55},
    {"id": "laptev",    "name": "拉普捷夫海","sector": "Atlantic","sector_cn": "大西洋扇区", "match_idx": "NAO", "mean_area": 0.55},
    {"id": "greenland", "name": "格陵兰海","sector": "Atlantic","sector_cn": "大西洋扇区", "match_idx": "NAO", "mean_area": 0.48},
    {"id": "central",   "name": "中北冰洋","sector": "Core",    "sector_cn": "核心区",     "match_idx": "AO",  "mean_area": 3.20},
]

# Phase4 untuned (unified E7v1 hyperparams)
PHASE4_UNTUNED = {
    "bering":    {"exp_id": "E26",  "baseline_rmse": 0.1014, "match_aux": "PNA", "match_rmse": 0.1002, "mismatch_aux": "NAO", "mismatch_rmse": 0.0996},
    "chukchi":   {"exp_id": "E29",  "baseline_rmse": 0.0803, "match_aux": "PNA", "match_rmse": 0.0807, "mismatch_aux": "NAO", "mismatch_rmse": 0.0815},
    "barents":   {"exp_id": "E27",  "baseline_rmse": 0.1000, "match_aux": "NAO", "match_rmse": 0.0983, "mismatch_aux": "PNA", "mismatch_rmse": 0.0995},
    "kara":      {"exp_id": "E30",  "baseline_rmse": 0.1171, "match_aux": "NAO", "match_rmse": 0.1188, "mismatch_aux": "PNA", "mismatch_rmse": 0.1175},
    "laptev":    {"exp_id": "E31",  "baseline_rmse": 0.1157, "match_aux": "NAO", "match_rmse": 0.1162, "mismatch_aux": "PNA", "mismatch_rmse": 0.1147},
    "greenland": {"exp_id": "E32",  "baseline_rmse": 0.0709, "match_aux": "NAO", "match_rmse": 0.0721, "mismatch_aux": "PNA", "mismatch_rmse": 0.0720},
    "central":   {"exp_id": "E28",  "baseline_rmse": 0.1435, "match_aux": "AO",  "match_rmse": 0.1435, "mismatch_aux": "SST", "mismatch_rmse": 0.1448},
}

# Phase4 independently tuned (Optuna 20-trial per region)
PHASE4_TUNED = {
    "bering":    {"best_aux": "NAO", "baseline_rmse": 0.1014, "tuned_rmse": 0.0926, "delta_pct": -8.6,  "lr": 0.00356, "aux_hidden": 48,  "aux_seq_len": 10, "aux_dropout": 0.8},
    "chukchi":   {"best_aux": "PNA", "baseline_rmse": 0.0803, "tuned_rmse": 0.0799, "delta_pct": -0.4,  "lr": 0.00424, "aux_hidden": 64,  "aux_seq_len": 10, "aux_dropout": 0.8},
    "barents":   {"best_aux": "NAO", "baseline_rmse": 0.1000, "tuned_rmse": 0.0927, "delta_pct": -7.3,  "lr": 0.00703, "aux_hidden": 80,  "aux_seq_len": 4,  "aux_dropout": 0.1},
    "kara":      {"best_aux": "PNA", "baseline_rmse": 0.1171, "tuned_rmse": 0.1055, "delta_pct": -9.9,  "lr": 0.00966, "aux_hidden": 48,  "aux_seq_len": 10, "aux_dropout": 0.4},
    "laptev":    {"best_aux": "PNA", "baseline_rmse": 0.1157, "tuned_rmse": 0.1191, "delta_pct": +3.3,  "lr": 0.00278, "aux_hidden": 64,  "aux_seq_len": 8,  "aux_dropout": 0.3},
    "greenland": {"best_aux": "PNA", "baseline_rmse": 0.0709, "tuned_rmse": 0.0695, "delta_pct": -2.0,  "lr": 0.00113, "aux_hidden": 112, "aux_seq_len": 4,  "aux_dropout": 0.1},
    "central":   {"best_aux": "AO",  "baseline_rmse": 0.1435, "tuned_rmse": 0.1426, "delta_pct": -0.6,  "lr": 0.00256, "aux_hidden": 128, "aux_seq_len": 8,  "aux_dropout": 0.3},
}

# Unified E7v1 hyperparams for comparison
UNIFIED_PARAMS = {"lr": 0.000635, "aux_seq_len": 6, "aux_dropout": 0.6}

# ===========================================================================
# Legacy baselines (from summary.py, for LR/RNN/LSTM comparison)
# ===========================================================================
LEGACY_BASELINES = {
    "LR":  {"label": "线性回归 (LR)","rmse": 0.5317, "params": 156},
    "RNN": {"label": "SimpleRNN",    "rmse": 0.5255, "params": "~5K"},
    "E1_old": {"label": "单变量 LSTM (E1)","rmse": 0.4975, "params": "~18K"},
}

# ===========================================================================
# All pan-Arctic experiments sorted by RMSE (for ranking chart)
# ===========================================================================
PAN_ARCTIC_RANKING = sorted(
    [{"id": k, **v} for k, v in {**PHASE1_SINGLE_VAR, **PHASE2_COMBINATIONS}.items()],
    key=lambda x: x["rmse"]
)
