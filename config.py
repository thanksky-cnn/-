# config.py
import os

# ==================== 路径配置 ====================
BASE_DIR = r"C:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE"
DATA_DIR = r"C:\Users\86152\PycharmProjects\2 +ao arctic_seaice_prediction lstm SIE\data\raw"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 输出子目录
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# ==================== 预测方案选择 ====================
# 可选: "short"(输出1月) | "medium"(输出6月) | "long"(输出12月)
PREDICTION_SCHEME = "long"

# ==================== 数据配置 ====================
TARGET_COLUMN = "area"  # "area" 或 "extent"

# 时间范围（None表示使用全部数据）
START_YEAR = None
END_YEAR = None

# ==================== 序列配置 ====================
INPUT_LEN = 12  # 用过去12个月预测（捕捉1年季节性模式）

# ==================== 各预测方案的专属参数 ====================
# Optuna 调优后会更新 "short" 和 "medium" 的值
_SCHEME_PARAMS = {

    "short": {  # 输入12月 → 输出1月（短期预测）
        "OUTPUT_LEN": 1,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.05686954819747893,
        "BATCH_SIZE": 16,
        "LEARNING_RATE": 0.000152601726333131,
        "WEIGHT_DECAY": 4.040318202658024e-05,
    },

    "medium": {  # 输入12月 → 输出6月（中期预测）
        "OUTPUT_LEN": 6,
        "HIDDEN_SIZE": 256,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.00012201620304831017,
        "WEIGHT_DECAY": 0.00046242986593718306,
    },

    "long": {  # 输入12月 → 输出12月（长期预测, Optuna 调优）
        "OUTPUT_LEN": 12,
        "HIDDEN_SIZE": 256,
        "NUM_LAYERS": 1,
        "DROPOUT": 0.1,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 0.0006347770217988592,
        "WEIGHT_DECAY": 2.585150663014067e-06,
    },
}

# ==================== 通用训练配置 ====================
NUM_EPOCHS = 1000
GRAD_CLIP_NORM = 1.0
REDUCE_LR_PATIENCE = 30
REDUCE_LR_FACTOR = 0.7

# ==================== 其他 ====================
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 30

# ==================== 应用所选方案 ====================
_active_scheme = _SCHEME_PARAMS.get(PREDICTION_SCHEME, _SCHEME_PARAMS["long"])
for _key, _val in _active_scheme.items():
    globals()[_key] = _val

# ==================== 创建输出目录 ====================
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
