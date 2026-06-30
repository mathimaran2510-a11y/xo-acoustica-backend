import time
import joblib
import numpy as np
import librosa

from app.config import MODEL_PATH
from app.feature_extractor import extract_features

# ==========================================
# Load Trained Model
# ==========================================

model = joblib.load(MODEL_PATH)


def predict_stress(audio_path):
    """
    Predict stress from an audio file.

    Returns:
    - Audio Information
    - Acoustic Features
    - ML Prediction
    """

    start_time = time.time()

    # ==========================================
    # Load Audio
    # ==========================================

    signal, sample_rate = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=signal, sr=sample_rate)

    total_samples = len(signal)

    # ==========================================
    # Extract Features
    # ==========================================

    features = extract_features(audio_path)

    feature_vector = np.array([[
        features["pitch"],
        features["jitter"],
        features["shimmer"],
        features["hnr"],
        features["spectral_centroid"]
    ]])

    # ==========================================
    # Prediction
    # ==========================================

    prediction = model.predict(feature_vector)[0]

    probabilities = model.predict_proba(feature_vector)[0]

    non_stress_probability = float(probabilities[0] * 100)

    stress_probability = float(probabilities[1] * 100)

    stress_score = stress_probability

    if prediction == 1:
        prediction_label = "Stress"
        model_confidence = stress_probability
    else:
        prediction_label = "Non Stress"
        model_confidence = non_stress_probability

    # ==========================================
    # Stress Level
    # ==========================================

    if stress_score < 35:
        stress_level = "Low"

    elif stress_score < 70:
        stress_level = "Moderate"

    else:
        stress_level = "High"

    processing_time = time.time() - start_time

    # ==========================================
    # Final Response
    # ==========================================

    return {

        # Model Information
        "algorithm": "Random Forest",
        "model_version": "1.0",
        "features_used": 5,

        # Audio Information
        "sample_rate": sample_rate,
        "duration": round(duration, 2),
        "total_samples": total_samples,
        "processing_time": round(processing_time, 3),

        # Acoustic Features
        "pitch": features["pitch"],
        "jitter": features["jitter"],
        "shimmer": features["shimmer"],
        "hnr": features["hnr"],
        "spectral_centroid": features["spectral_centroid"],

        # ML Prediction
        "prediction": prediction_label,
        "stress_score": round(stress_score, 2),
        "stress_level": stress_level,
        "model_confidence": round(model_confidence, 2),

        # Class Probabilities
        "probability_non_stress": round(non_stress_probability, 2),
        "probability_stress": round(stress_probability, 2)
    }