from pathlib import Path

from predictor import predict_stress

audio = Path(
    "../dataset/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav"
)

result = predict_stress(audio)

print("\n========== Prediction ==========\n")

for key, value in result.items():
    print(f"{key:20} : {value}")