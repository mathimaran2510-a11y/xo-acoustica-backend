from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# DATASET
# (Only used for training - not required during deployment)
# ==========================================================

DATASET_DIR = BASE_DIR / "dataset" / "RAVDESS"

# ==========================================================
# CSV DIRECTORY
# ==========================================================

CSV_DIR = BASE_DIR / "csv"
CSV_DIR.mkdir(exist_ok=True)

TRAINING_CSV = CSV_DIR / "training.csv"

# ==========================================================
# MODEL DIRECTORY
# ==========================================================

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "stress_model.pkl"

# ==========================================================
# SUPPORTED AUDIO FORMATS
# ==========================================================

SUPPORTED_AUDIO = [
    ".wav",
    ".mp3",
    ".m4a"
]

# ==========================================================
# MACHINE LEARNING FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    "pitch",
    "jitter",
    "shimmer",
    "hnr",
    "spectral_centroid"
]

# ==========================================================
# RANDOM FOREST SETTINGS
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ==========================================================
# APPLICATION INFORMATION
# ==========================================================

APP_NAME = "XO Acoustica"

MODEL_NAME = "Random Forest"

MODEL_VERSION = "1.0"

API_VERSION = "1.0"