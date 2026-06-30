import librosa
import numpy as np
import parselmouth

from parselmouth.praat import call


def extract_features(audio_path):
    """
    Extract:
        Pitch
        Jitter
        Shimmer
        HNR
        Spectral Centroid
    """

    # -----------------------
    # Load audio using librosa
    # -----------------------
    signal, sr = librosa.load(audio_path, sr=None)

    # -----------------------
    # Spectral Centroid
    # -----------------------
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    # -----------------------
    # Parselmouth
    # -----------------------
    sound = parselmouth.Sound(str(audio_path))

    # -----------------------
    # Pitch
    # -----------------------
    pitch = call(sound, "To Pitch", 0.0, 75, 600)

    pitch_values = pitch.selected_array["frequency"]

    pitch_values = pitch_values[pitch_values != 0]

    if len(pitch_values) == 0:
        mean_pitch = 0

    else:
        mean_pitch = np.mean(pitch_values)

    # -----------------------
    # Point Process
    # -----------------------
    point_process = call(
        sound,
        "To PointProcess (periodic, cc)",
        75,
        600
    )

    # -----------------------
    # Jitter
    # -----------------------
    jitter = call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3
    )

    # -----------------------
    # Shimmer
    # -----------------------
    shimmer = call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6
    )

    # -----------------------
    # Harmonicity
    # -----------------------
    harmonicity = call(
        sound,
        "To Harmonicity (cc)",
        0.01,
        75,
        0.1,
        1.0
    )

    hnr = call(
        harmonicity,
        "Get mean",
        0,
        0
    )

    return {
        "pitch": round(float(mean_pitch), 2),
        "jitter": round(float(jitter), 5),
        "shimmer": round(float(shimmer), 5),
        "hnr": round(float(hnr), 2),
        "spectral_centroid": round(float(spectral_centroid), 2)
    }