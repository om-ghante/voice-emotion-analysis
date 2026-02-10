import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

def extract_mfcc(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, n_mfcc: int = 13) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs

def extract_mel_spectrogram(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, n_mels: int = 80) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_spectral_features(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)[0]

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
    }

def extract_energy(waveform: np.ndarray) -> float:
    rms = np.sqrt(np.mean(waveform ** 2))
    return float(rms)

def extract_zero_crossing_rate(waveform: np.ndarray) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    return zcr

def extract_all_features(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    mfcc = extract_mfcc(waveform, sample_rate)
    mel_spec = extract_mel_spectrogram(waveform, sample_rate)
    spectral = extract_spectral_features(waveform, sample_rate)
    energy = extract_energy(waveform)
    zcr = extract_zero_crossing_rate(waveform)

    return {
        "mfcc": mfcc,
        "mel_spectrogram": mel_spec,
        "spectral_centroid": spectral["spectral_centroid"],
        "spectral_bandwidth": spectral["spectral_bandwidth"],
        "spectral_rolloff": spectral["spectral_rolloff"],
        "rms_energy": energy,
        "zero_crossing_rate": zcr,
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "mfcc_std": np.std(mfcc, axis=1).tolist(),
    }
