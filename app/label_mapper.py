"""
label_mapper.py

Converts RAVDESS emotion labels into
Stress / Non-Stress labels.
"""

# RAVDESS Emotion Codes
# 01 = Neutral
# 02 = Calm
# 03 = Happy
# 04 = Sad
# 05 = Angry
# 06 = Fearful
# 07 = Disgust
# 08 = Surprised

EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

# Binary stress mapping
#
# 0 = Non-Stress
# 1 = Stress
#
# We exclude:
# Disgust
# Surprised

STRESS_MAP = {
    "01": 0,   # Neutral
    "02": 0,   # Calm
    "03": 0,   # Happy

    "04": 1,   # Sad
    "05": 1,   # Angry
    "06": 1    # Fearful
}


def get_emotion(filename):
    """
    Returns emotion name from filename.
    """

    emotion_code = filename.split("-")[2]

    return EMOTION_MAP.get(emotion_code)


def get_stress_label(filename):
    """
    Returns:

    0 = Non Stress
    1 = Stress
    None = Ignore
    """

    emotion_code = filename.split("-")[2]

    return STRESS_MAP.get(emotion_code)