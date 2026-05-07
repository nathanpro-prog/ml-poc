from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Engine 1 — Match outcome prediction (WL_BIN)
# Train with: python src/train.py --engine 1
MODELS = {
    "engine1_logistic_regression": {
        "name": "Logistic Regression (Engine 1)",
        "description": "Match outcome prediction — linear baseline with StandardScaler.",
        "path": MODELS_DIR / "engine1" / "logistic_regression.pkl",
    },
    "engine1_random_forest": {
        "name": "Random Forest (Engine 1)",
        "description": "Match outcome prediction — ensemble with feature importance.",
        "path": MODELS_DIR / "engine1" / "random_forest.pkl",
    },
    "engine1_xgboost": {
        "name": "XGBoost (Engine 1)",
        "description": "Match outcome prediction — gradient boosting, best expected performance.",
        "path": MODELS_DIR / "engine1" / "xgboost.pkl",
    },
}