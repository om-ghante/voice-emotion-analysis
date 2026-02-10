import os
import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_DURATION = 2.0
TARGET_SAMPLE_RATE = 16000

def load_audio(filepath: str, target_sr: int = TARGET_SAMPLE_RATE) -> tuple:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".wav", ".mp3"):
        raise ValueError(f"Unsupported format: {ext}")

    tmp_wav = None
    load_path = filepath
    
    if ext == ".mp3":
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(filepath)
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir())
            audio.export(tmp_wav.name, format="wav")
            load_path = tmp_wav.name
            tmp_wav.close()
        except ImportError:
            pass
        except Exception:
            load_path = filepath

    try:
        waveform, sr = librosa.load(load_path, sr=target_sr, mono=True)
        return waveform, sr
    finally:
        if tmp_wav and os.path.exists(tmp_wav.name):
            os.remove(tmp_wav.name)

def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform

def chunk_audio(filepath: str, chunk_duration: float = DEFAULT_CHUNK_DURATION, target_sr: int = TARGET_SAMPLE_RATE) -> list:
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

    return chunks

def save_chunk_to_wav(chunk: dict, output_path: str) -> str:
    sf.write(output_path, chunk["waveform"], chunk["sample_rate"])
    return output_path

def get_audio_duration(filepath: str) -> float:
    return round(librosa.get_duration(path=filepath), 2)
