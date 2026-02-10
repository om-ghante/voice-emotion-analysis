import os
import torch
import torchaudio
import logging
from speechbrain.inference.interfaces import foreign_class

logger = logging.getLogger(__name__)

_model = None
_MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "emotion-recognition")

EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]

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
    global _model
    if _model is None:
        os.makedirs(_SAVEDIR, exist_ok=True)
        _model = foreign_class(
            source=_MODEL_SOURCE,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=_SAVEDIR,
        )
    return _model

def predict_emotion(audio_path: str) -> dict:
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
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir()) as tmp:
        tmp_path = tmp.name
        torchaudio.save(tmp_path, waveform, sample_rate)

    try:
        result = predict_emotion(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result
