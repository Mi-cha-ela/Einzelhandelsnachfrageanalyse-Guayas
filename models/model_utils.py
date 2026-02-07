import json
from pathlib import Path
import mlflow
import mlflow.pyfunc

def load_feature_cols(feature_cols_path: Path) -> list[str]:
    with open(feature_cols_path, "r") as f:
        return json.load(f)

def load_mlflow_model(run_id: str, tracking_uri: str):
    """
    Lädt das Modell, das in MLflow im Run gespeichert wurde.
    Wir nehmen an: im Training wurde geloggt mit
    mlflow.sklearn.log_model(model, name="model")
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    return mlflow.pyfunc.load_model(model_uri)

