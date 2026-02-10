"""
Audio Segmentation Module.
Loads audio files, normalizes them, and splits into fixed-length chunks
with exact start and end timestamps.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

# Default chunk duration in seconds
DEFAULT_CHUNK_DURATION = 2.0
# Target sample rate for processing
TARGET_SAMPLE_RATE = 16000


def load_audio(filepath: str, target_sr: int = TARGET_SAMPLE_RATE) -> tuple:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        filepath: Path to the audio file (.wav or .mp3).
        target_sr: Target sample rate.

    Returns:
        Tuple of (waveform as numpy array, sample rate).

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".wav", ".mp3"):
        raise ValueError(f"Unsupported audio format: {ext}. Use .wav or .mp3.")

    logger.info(f"Loading audio file: {filepath}")
    waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)
    logger.info(f"Audio loaded: {len(waveform)} samples at {sr} Hz, "
                f"duration: {len(waveform) / sr:.2f}s")
    return waveform, sr


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize audio waveform to [-1, 1] range.

    Args:
        waveform: Input audio waveform.

    Returns:
        Normalized waveform.
    """
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def chunk_audio(
    filepath: str,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> list:
    """
    Load, normalize, and split audio into fixed-length chunks.

    Each chunk includes the waveform data and exact start/end timestamps.

    Args:
        filepath: Path to the audio file.
        chunk_duration: Duration of each chunk in seconds.
        target_sr: Target sample rate.

    Returns:
        List of dictionaries, each containing:
            - 'waveform': numpy array of the chunk
            - 'sample_rate': sample rate
            - 'start_sec': start time in seconds
            - 'end_sec': end time in seconds
            - 'chunk_index': integer index of the chunk
    """
    waveform, sr = load_audio(filepath, target_sr)
    waveform = normalize_audio(waveform)

    total_duration = len(waveform) / sr
    chunk_samples = int(chunk_duration * sr)
    total_samples = len(waveform)

    chunks = []
    chunk_index = 0
    offset = 0

    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk_waveform = waveform[offset:end]

        # Skip very short chunks (less than 0.5 seconds)
        chunk_len_sec = len(chunk_waveform) / sr
        if chunk_len_sec < 0.5:
            break

        start_sec = offset / sr
        end_sec = end / sr

        chunks.append({
            "waveform": chunk_waveform,
            "sample_rate": sr,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "chunk_index": chunk_index,
        })

        chunk_index += 1
        offset = end

    logger.info(f"Audio split into {len(chunks)} chunks of ~{chunk_duration}s each. "
                f"Total duration: {total_duration:.2f}s")
    return chunks


def save_chunk_to_wav(chunk: dict, output_path: str) -> str:
    """
    Save a single chunk to a .wav file on disk.

    Args:
        chunk: Chunk dictionary from chunk_audio().
        output_path: Path for the output .wav file.

    Returns:
        The output path.
    """
    sf.write(output_path, chunk["waveform"], chunk["sample_rate"])
    return output_path


def get_audio_duration(filepath: str) -> float:
    """
    Get the total duration of an audio file in seconds.

    Args:
        filepath: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    duration = librosa.get_duration(filename=filepath)
    return round(duration, 2)
