"""
Audio feature extraction — mirrors the training notebook exactly.
Produces a 624-dim vector (tabular) and a (128,128,1) mel-spectrogram image.
Both functions accept a raw numpy audio array instead of a file path.
"""
import numpy as np
import librosa
from PIL import Image

SR = 22050
DURATION = 3


def extract_features(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return a 624-dimensional float32 feature vector for one audio chunk."""
    target_len = sr * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    feats = []

    # 1. MFCCs — 80 dims
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    feats.extend(np.mean(mfccs.T, axis=0))
    feats.extend(np.std(mfccs.T, axis=0))

    # 2. MFCC delta — 80 dims
    d1 = librosa.feature.delta(mfccs)
    feats.extend(np.mean(d1.T, axis=0))
    feats.extend(np.std(d1.T, axis=0))

    # 3. MFCC delta-delta — 80 dims
    d2 = librosa.feature.delta(mfccs, order=2)
    feats.extend(np.mean(d2.T, axis=0))
    feats.extend(np.std(d2.T, axis=0))

    # 4. MFCC delta covariance diagonal — 40 dims
    feats.extend(np.diag(np.cov(d1)))

    # 5. Chroma — 24 dims
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.extend(np.mean(chroma.T, axis=0))
    feats.extend(np.std(chroma.T, axis=0))

    # 6. Mel spectrogram — 256 dims
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    feats.extend(np.mean(mel.T, axis=0))
    feats.extend(np.std(mel.T, axis=0))

    # 7. Log-mel energy bands — 16 dims
    mel_log = librosa.power_to_db(mel)
    band_size = mel_log.shape[0] // 8
    for b in range(8):
        band = mel_log[b * band_size:(b + 1) * band_size, :]
        feats.append(float(np.mean(band)))
        feats.append(float(np.std(band)))

    # 8. Spectral contrast — 14 dims
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    feats.extend(np.mean(contrast.T, axis=0))
    feats.extend(np.std(contrast.T, axis=0))

    # 9. Tonnetz — 12 dims
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    feats.extend(np.mean(tonnetz.T, axis=0))
    feats.extend(np.std(tonnetz.T, axis=0))

    # 10. Pitch / F0 — 4 dims
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_clean = np.where(np.isnan(f0), 0.0, f0)
    voiced = f0_clean[f0_clean > 0]
    feats.append(float(np.mean(voiced)) if len(voiced) > 0 else 0.0)
    feats.append(float(np.std(voiced)) if len(voiced) > 0 else 0.0)
    feats.append(float(np.ptp(voiced)) if len(voiced) > 0 else 0.0)
    feats.append(float(np.mean(voiced_flag.astype(float))))

    # 11. Harmonic / Percussive — 4 dims
    harmonic, percussive = librosa.effects.hpss(audio)
    h_rms = float(np.sqrt(np.mean(harmonic ** 2)))
    p_rms = float(np.sqrt(np.mean(percussive ** 2)))
    feats.append(h_rms)
    feats.append(p_rms)
    feats.append(h_rms / (p_rms + 1e-8))
    feats.append(h_rms / (h_rms + p_rms + 1e-8))

    # 12. Zero-Crossing Rate — 2 dims
    zcr = librosa.feature.zero_crossing_rate(audio)
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.std(zcr)))

    # 13. RMS Energy — 2 dims
    rms = librosa.feature.rms(y=audio)
    feats.append(float(np.mean(rms)))
    feats.append(float(np.std(rms)))

    # 14. Spectral Centroid — 2 dims
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    feats.append(float(np.mean(centroid)))
    feats.append(float(np.std(centroid)))

    # 15. Spectral Bandwidth — 2 dims
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    feats.append(float(np.mean(bandwidth)))
    feats.append(float(np.std(bandwidth)))

    # 16. Spectral Rolloff — 2 dims
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    feats.append(float(np.mean(rolloff)))
    feats.append(float(np.std(rolloff)))

    # 17. Poly Features — 4 dims
    poly = librosa.feature.poly_features(y=audio, sr=sr, order=1)
    feats.extend(np.mean(poly.T, axis=0))
    feats.extend(np.std(poly.T, axis=0))

    result = np.array(feats, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def extract_melspec_image(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return a (128, 128, 1) float32 mel-spectrogram image for the CNN branch."""
    target_len = sr * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = Image.fromarray(mel_db).resize((128, 128), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32)
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)

    return img_arr[..., np.newaxis]  # (128, 128, 1)
