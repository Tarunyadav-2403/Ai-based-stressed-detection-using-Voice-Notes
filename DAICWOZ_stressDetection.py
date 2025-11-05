import os
import librosa
import numpy as np
import pandas as pd

# -------------------------------
# 1. Paths
# -------------------------------
audio_dir = r"C:\CU\7th Sem\Capstone Project\Research Papers\DAIC-WOZ Datasets\Audio Features"
output_csv = r"C:\CU\7th Sem\Capstone Project\Research Papers\DAIC-WOZ datasets\stress_ratings_diverse.csv"

# -------------------------------
# 2. Check folder exists
# -------------------------------
if not os.path.exists(audio_dir):
    raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

# -------------------------------
# 3. Select target files 300-324
# -------------------------------
wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
target_files = [f for f in wav_files if f.split("_")[0].isdigit() and 300 <= int(f.split("_")[0]) <= 324]

if not target_files:
    raise FileNotFoundError("No WAV files found from 300_AUDIO to 324_AUDIO")

print(f"Found {len(target_files)} target WAV files.")

# -------------------------------
# 4. Helper: Stress interpretation
# -------------------------------
def interpret_stress(score):
    if score == 0:
        return "Low Stress"
    elif score == 1:
        return "Low-moderate Stress"
    elif score == 2:
        return "Moderate Stress"
    elif score == 3:
        return "High Stress"
    else:
        return "Very High Stress"

# -------------------------------
# 5. Extract features
# -------------------------------
features_list = []

for f in target_files:
    file_path = os.path.join(audio_dir, f)
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0 or np.isnan(y).any():
            print(f"Skipping {f} due to empty/invalid audio")
            continue

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc))
        mfcc_std = float(np.mean(np.std(mfcc, axis=1)))

        # Energy
        rms = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # Pitch
        try:
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[mags > np.median(mags)]
            pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0
            pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0
        except:
            pitch_mean = 0
            pitch_std = 0

        # Speech rate (beats/sec)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        speech_rate = float(tempo / 60)

        features_list.append({
            "file": f,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "speech_rate": speech_rate
        })

        print(f"Extracted features from {f}")

    except Exception as e:
        print(f"Failed {f}: {e}")


# 6. Convert to DataFrame
df = pd.DataFrame(features_list)

# 7. Dynamic normalization

def normalize(col):
    return (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else np.zeros_like(col)

df["pitch_score"] = normalize(df["pitch_mean"]) * 0.4 + normalize(df["pitch_std"]) * 0.1
df["energy_score"] = normalize(df["energy_mean"]) * 0.2 + normalize(df["energy_std"]) * 0.05
df["mfcc_score"] = normalize(df["mfcc_mean"]) * 0.15 + normalize(df["mfcc_std"]) * 0.05
df["speech_score"] = normalize(df["speech_rate"]) * 0.25

# 8. Compute combined stress score
df["stress_raw"] = df["pitch_score"] + df["energy_score"] + df["mfcc_score"] + df["speech_score"]
df["stress_score"] = ((df["stress_raw"] / df["stress_raw"].max()) * 4 + 1).round().astype(int)  # maps 0-max to 1-5
df["stress_score"] = df["stress_score"].clip(1,5)
df["stress_description"] = df["stress_score"].apply(interpret_stress)

# 9. Save CSV
df.to_csv(output_csv, index=False)
print(f"\nStress ratings saved to: {output_csv}")
