"""
Audio Feature Extraction Module.
Provides feature extraction utilities compatible with SpeechBrain input format.
Extracts MFCCs, spectral features, and energy for audio analysis.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def extract_mfcc(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE,
                 n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from a waveform.

    Args:
        waveform: Audio waveform as a numpy array.
        sample_rate: Sample rate of the audio.
        n_mfcc: Number of MFCC coefficients to extract.

    Returns:
        MFCC feature matrix (n_mfcc x time_frames).
    """
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    logger.debug(f"Extracted MFCC features: shape={mfccs.shape}")
    return mfccs


def extract_mel_spectrogram(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE,
                            n_mels: int = 80) -> np.ndarray:
    """
    Extract Mel spectrogram from a waveform.

    Args:
        waveform: Audio waveform as a numpy array.
        sample_rate: Sample rate of the audio.
        n_mels: Number of Mel bands.

    Returns:
        Mel spectrogram matrix (n_mels x time_frames).
    """
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    logger.debug(f"Extracted Mel spectrogram: shape={mel_spec_db.shape}")
    return mel_spec_db


def extract_spectral_features(waveform: np.ndarray,
                              sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    """
    Extract spectral features including centroid, bandwidth, and rolloff.

    Args:
        waveform: Audio waveform as a numpy array.
        sample_rate: Sample rate of the audio.

    Returns:
        Dictionary with spectral feature arrays.
    """
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)[0]

    features = {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
    }
    logger.debug("Extracted spectral features.")
    return features


def extract_energy(waveform: np.ndarray) -> float:
    """
    Compute the RMS energy of a waveform.

    Args:
        waveform: Audio waveform as a numpy array.

    Returns:
        RMS energy value as a float.
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    return float(rms)


def extract_zero_crossing_rate(waveform: np.ndarray) -> np.ndarray:
    """
    Extract zero crossing rate from a waveform.

    Args:
        waveform: Audio waveform as a numpy array.

    Returns:
        Zero crossing rate array.
    """
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    return zcr


def extract_all_features(waveform: np.ndarray,
                         sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    """
    Extract a comprehensive set of audio features from a waveform.

    Args:
        waveform: Audio waveform as a numpy array.
        sample_rate: Sample rate of the audio.

    Returns:
        Dictionary containing all extracted features.
    """
    mfcc = extract_mfcc(waveform, sample_rate)
    mel_spec = extract_mel_spectrogram(waveform, sample_rate)
    spectral = extract_spectral_features(waveform, sample_rate)
    energy = extract_energy(waveform)
    zcr = extract_zero_crossing_rate(waveform)

    features = {
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

    logger.info(f"Extracted all features: energy={energy:.4f}, "
                f"mfcc_shape={mfcc.shape}")
    return features
