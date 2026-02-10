"""
Core Emotion Analysis Pipeline Service.
Orchestrates the entire analysis flow: chunking, prediction, merging, and formatting.
"""

import os
import logging
import tempfile
import soundfile as sf
from audio_processing.chunker import chunk_audio, get_audio_duration
from models.emotion_model import predict_emotion
from utils.time_utils import seconds_to_mmss, format_duration, format_segment

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str, chunk_duration: float = 2.0) -> dict:
    """
    Run the full emotion analysis pipeline on an audio file.

    Steps:
        1. Validate audio file exists
        2. Get total duration
        3. Chunk audio into segments
        4. Predict emotion for each chunk
        5. Merge consecutive identical emotions
        6. Format timestamps to MM:SS

    Args:
        audio_path: Path to the audio file.
        chunk_duration: Duration of each analysis chunk in seconds.

    Returns:
        Dictionary containing:
            - 'duration': total duration as MM:SS string
            - 'duration_seconds': total duration in seconds
            - 'segments': list of emotion segments with start, end, emotion
            - 'raw_segments': list of per-chunk results before merging
            - 'summary': emotion distribution summary
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Step 1: Get total duration
    total_duration = get_audio_duration(audio_path)
    logger.info(f"Analyzing audio: {audio_path} (duration: {total_duration:.2f}s)")

    # Step 2: Chunk audio
    chunks = chunk_audio(audio_path, chunk_duration=chunk_duration)
    logger.info(f"Created {len(chunks)} chunks")

    if not chunks:
        return {
            "duration": format_duration(total_duration),
            "duration_seconds": total_duration,
            "segments": [],
            "raw_segments": [],
            "summary": {},
        }

    # Step 3: Predict emotion for each chunk
    raw_segments = []
    for chunk in chunks:
        # Save chunk to temporary wav for SpeechBrain inference
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir()) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, chunk["waveform"], chunk["sample_rate"])

        try:
            result = predict_emotion(tmp_path)
            raw_segments.append({
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "scores": result["scores"],
                "chunk_index": chunk["chunk_index"],
            })
            logger.debug(f"Chunk {chunk['chunk_index']}: "
                         f"{chunk['start_sec']:.2f}s - {chunk['end_sec']:.2f}s -> "
                         f"{result['emotion']} ({result['confidence']:.4f})")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Step 4: Merge consecutive identical emotions
    merged_segments = _merge_consecutive_emotions(raw_segments)

    # Step 5: Format segments with MM:SS timestamps
    formatted_segments = []
    for seg in merged_segments:
        formatted_segments.append(format_segment(
            seg["start_sec"], seg["end_sec"], seg["emotion"]
        ))

    # Step 6: Compute emotion distribution summary
    summary = _compute_emotion_summary(raw_segments, total_duration)

    return {
        "duration": format_duration(total_duration),
        "duration_seconds": total_duration,
        "segments": formatted_segments,
        "raw_segments": [
            {
                "start": seconds_to_mmss(s["start_sec"]),
                "end": seconds_to_mmss(s["end_sec"]),
                "emotion": s["emotion"],
                "confidence": s["confidence"],
            }
            for s in raw_segments
        ],
        "summary": summary,
    }


def _merge_consecutive_emotions(segments: list) -> list:
    """
    Merge consecutive segments that share the same emotion label.

    Args:
        segments: List of raw segment dictionaries.

    Returns:
        List of merged segment dictionaries.
    """
    if not segments:
        return []

    merged = []
    current = {
        "start_sec": segments[0]["start_sec"],
        "end_sec": segments[0]["end_sec"],
        "emotion": segments[0]["emotion"],
    }

    for seg in segments[1:]:
        if seg["emotion"] == current["emotion"]:
            current["end_sec"] = seg["end_sec"]
        else:
            merged.append(current.copy())
            current = {
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "emotion": seg["emotion"],
            }

    merged.append(current.copy())
    logger.info(f"Merged {len(segments)} chunks into {len(merged)} segments")
    return merged


def _compute_emotion_summary(raw_segments: list, total_duration: float) -> dict:
    """
    Compute emotion distribution as percentage of total time.

    Args:
        raw_segments: List of per-chunk segment dictionaries.
        total_duration: Total audio duration in seconds.

    Returns:
        Dictionary mapping emotion labels to their percentage of total time.
    """
    if not raw_segments or total_duration == 0:
        return {}

    emotion_time = {}
    for seg in raw_segments:
        duration = seg["end_sec"] - seg["start_sec"]
        emotion = seg["emotion"]
        emotion_time[emotion] = emotion_time.get(emotion, 0.0) + duration

    summary = {}
    for emotion, time_val in emotion_time.items():
        percentage = round((time_val / total_duration) * 100, 1)
        summary[emotion] = {
            "duration_seconds": round(time_val, 2),
            "percentage": percentage,
        }

    return summary
