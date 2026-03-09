import joblib
import re
from pathlib import Path

MODELS_PATH = Path("ml/models")

_type_model = None

def _load_type_model():
    global _type_model
    if _type_model is None:
        path = MODELS_PATH / "type_classifier.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Type classifier not found at {path}. "
                "Run model_training.py first."
            )
        _type_model = joblib.load(path)
    return _type_model

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\d+", "NUM", text) 
    return text

def predict_type(title: str, description: str) -> str:
    model = _load_type_model()
    text = "[TITLE] " + title + " [DESC] " + description
    text = normalize_text(text)

    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = model.classes_

    scores = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}
    confidence = round(float(max(proba)), 3)

    return {
        "label": label,
        "confidence": confidence,
        "scores": scores
    }
    
    

