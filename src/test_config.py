from App.config import BASE_DIR, MLFLOW_TRACKING_URI, DATA_DIR, MODELS_DIR, REPORTS_DIR
import os

print("BASE_DIR:", BASE_DIR)
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
print("DATA_DIR exists:", os.path.exists(DATA_DIR), DATA_DIR)
print("MODELS_DIR exists:", os.path.exists(MODELS_DIR), MODELS_DIR)
print("REPORTS_DIR exists:", os.path.exists(REPORTS_DIR), REPORTS_DIR)