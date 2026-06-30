from pathlib import Path

from feature_extractor import extract_features

# Change this to one WAV file from your RAVDESS dataset
audio_file = Path(
    "../dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
)

features = extract_features(audio_file)

print("\nExtracted Features\n")
print("----------------------------")

for key, value in features.items():
    print(f"{key:20} : {value}")