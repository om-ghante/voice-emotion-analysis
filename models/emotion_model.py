"""
Emotion Prediction Engine using SpeechBrain pretrained model.
Loads a pretrained Speech Emotion Recognition model and provides
a reusable prediction function for audio chunks.
"""

import os
import torch
import torchaudio
import logging
from speechbrain.inference.interfaces import foreign_class

logger = logging.getLogger(__name__)

# Singleton model instance
_model = None
_MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "emotion-recognition")

# Emotion label mapping from the IEMOCAP model
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]

# Map abbreviated model labels to full names
LABEL_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "neutral": "neutral",
    "happy": "happy",
    "angry": "angry",
}


def _get_model():
    """
    Load and cache the SpeechBrain emotion recognition model.
    Uses a singleton pattern to avoid reloading on every call.

    Returns:
        The loaded SpeechBrain classifier instance.
    """
    global _model
    if _model is None:
        logger.info("Loading SpeechBrain emotion recognition model...")
        os.makedirs(_SAVEDIR, exist_ok=True)
        _model = foreign_class(
            source=_MODEL_SOURCE,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=_SAVEDIR,
        )
        logger.info("Emotion recognition model loaded successfully.")
    return _model


def predict_emotion(audio_path: str) -> dict:
    """
    Predict emotion from an audio file.

    Args:
        audio_path: Path to a .wav audio file.

    Returns:
        Dictionary with:
            - 'emotion': predicted emotion label (str)
            - 'confidence': confidence score (float)
            - 'scores': dict mapping each emotion label to its score
    """
    model = _get_model()

    out_prob, score, index, text_lab = model.classify_file(audio_path)

    probabilities = out_prob.squeeze().tolist()
    predicted_label = text_lab[0].lower()
    predicted_label = LABEL_MAP.get(predicted_label, predicted_label)
    confidence = score.item()

    scores = {}
    for i, label in enumerate(EMOTION_LABELS):
        if i < len(probabilities):
            scores[label] = round(probabilities[i], 4)

    return {
        "emotion": predicted_label,
        "confidence": round(confidence, 4),
        "scores": scores,
    }


def predict_emotion_from_waveform(waveform: torch.Tensor, sample_rate: int) -> dict:
    """
    Predict emotion from a waveform tensor by saving to a temporary file.

    Args:
        waveform: Audio waveform as a torch Tensor (channels x samples).
        sample_rate: Sample rate of the waveform.

    Returns:
        Dictionary with emotion, confidence, and scores.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        torchaudio.save(tmp_path, waveform, sample_rate)

    try:
        result = predict_emotion(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result
