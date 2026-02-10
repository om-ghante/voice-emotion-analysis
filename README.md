# Voice Emotion Analysis

A production-ready web application that analyzes voice/audio input to detect emotions over time and visualizes emotional changes.

## Features

- **Audio Upload**: Upload WAV or MP3 files for analysis
- **Time-Based Emotion Detection**: Audio is split into 2-second chunks with precise timestamps
- **Emotion Recognition**: Uses SpeechBrain's pretrained wav2vec2-IEMOCAP model
- **Emotion Timeline**: Start time, end time, and detected emotion for each segment
- **Visual Interface**: Professional minimal design with Navy Blue theme

## Technology Stack

- **Backend**: Flask, Gunicorn
- **ML Model**: SpeechBrain (wav2vec2-IEMOCAP)
- **Audio Processing**: Librosa, Torchaudio, SoundFile, Pydub, FFmpeg
- **Frontend**: HTML5, CSS3, JavaScript

## Setup and Installation

### Prerequisites

- Python 3.10+
- FFmpeg (required for MP3 support)

### Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python app.py
```
Or with Gunicorn:
```bash
gunicorn app:app --bind 0.0.0.0:5000
```

Open browser at `http://localhost:5000`

## Deployment (Render)

1. Create a Web Service on Render
2. Connect repository
3. Set Build Command: `apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt`
4. Set Start Command: `bash start.sh`

## API Reference

### POST `/upload`
- **Field**: `audio` (file)
- **Returns**: JSON with segments and summary
