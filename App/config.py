from pathlib import Path

# Projekt-Root = ein Ordner über App/
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
MLRUNS_DIR  = BASE_DIR / "mlruns"

# Für die App verwenden wir dein Q1-File
DATA_PATH = DATA_DIR / "guayas_top3_q1_2014.pkl"

# MLflow: wir nutzen local file store (mlruns/)
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR.resolve()}"


# Hier trägst du die Run-ID vom besten Lauf ein (aus MLflow UI)
RUN_ID = "e582a52422a0444db61ecb6f981f0cf3"
MODEL_URI = f"runs:/{RUN_ID}/model"
# Safety: Ordner erstellen falls nicht da
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)


print("CONFIG LOADED:", __file__)
print("RUN_ID:", RUN_ID)
