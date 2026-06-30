from pathlib import Path

import pandas as pd
from tqdm import tqdm

from app.config import DATASET_DIR, TRAINING_CSV
from app.feature_extractor import extract_features
from app.label_mapper import get_emotion, get_stress_label


def build_dataset():

    dataset = []

    wav_files = list(DATASET_DIR.rglob("*.wav"))

    print(f"\nFound {len(wav_files)} WAV files\n")

    for wav in tqdm(wav_files):

        label = get_stress_label(wav.name)

        if label is None:
            continue

        try:

            features = extract_features(wav)

            row = {
                "filename": wav.name,
                "emotion": get_emotion(wav.name),
                "stress": label,
                **features
            }

            dataset.append(row)

        except Exception as e:

            print(f"Skipped {wav.name}")

            print(e)

    df = pd.DataFrame(dataset)

    df.to_csv(TRAINING_CSV, index=False)

    print("\nDataset Saved Successfully")

    print(TRAINING_CSV)

    print(df.head())


if __name__ == "__main__":

    build_dataset()