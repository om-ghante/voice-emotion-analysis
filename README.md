# Voice Emotion Analysis Dashboard

A production-ready web application that analyzes voice/audio input to detect emotions over time and visualizes emotional changes on a professional dashboard.

The system accepts audio files (WAV/MP3), splits them into time-based segments, performs speech emotion recognition using a pretrained SpeechBrain model (wav2vec2-IEMOCAP), and presents the results through both a Flask web interface and a Streamlit analytics dashboard.

---

## Features

- **Audio Upload**: Upload WAV or MP3 files for analysis
- **Time-Based Emotion Detection**: Audio is split into 2-second chunks with precise timestamps
- **Emotion Recognition**: Uses SpeechBrain's pretrained wav2vec2-IEMOCAP model to classify emotions (angry, happy, neutral, sad)
- **Consecutive Merging**: Adjacent segments with the same emotion are merged for cleaner timelines
- **Emotion Timeline Table**: Start time, end time, and detected emotion for each segment
- **Emotion Over Time Chart**: Visual timeline showing emotion transitions
- **Emotion Distribution Chart**: Bar chart showing percentage of time in each emotion
- **Detailed Chunk Analysis**: Per-chunk results with confidence scores
- **Dual Interface**: Flask web UI and Streamlit analytics dashboard

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend API | Flask |
| ML Model | SpeechBrain (wav2vec2-IEMOCAP) |
| Audio Processing | Librosa, Torchaudio, SoundFile |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Language | Python 3.10 |

---

## Project Structure

```
.
├── app.py                        # Flask API server
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for deployment
├── README.md                     # Documentation
│
├── uploads/                      # Temporary upload storage
│   └── .gitkeep
│
├── models/
│   └── emotion_model.py          # SpeechBrain emotion prediction engine
│
├── audio_processing/
│   ├── chunker.py                # Audio segmentation with timestamps
│   └── features.py               # Audio feature extraction (MFCC, Mel, etc.)
│
├── services/
│   └── emotion_service.py        # Core analysis pipeline
│
├── dashboard/
│   └── dashboard.py              # Streamlit analytics dashboard
│
├── static/
│   └── style.css                 # Minimal CSS (black/white/gray)
│
├── templates/
│   └── index.html                # Flask upload interface
│
└── utils/
    └── time_utils.py             # Timestamp formatting utilities
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Voice Emotion Analysis"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The SpeechBrain model will download automatically on the first run (~300 MB).

---

## Running the Application

### Option 1: Streamlit Dashboard (Recommended)

Run the Streamlit dashboard directly. This does not require the Flask server for analysis.

```bash
streamlit run dashboard/dashboard.py --server.port 8501
```

Open your browser at: `http://localhost:8501`

### Option 2: Flask Web Interface

Start the Flask API server:

```bash
python app.py
```

Open your browser at: `http://localhost:5000`

### Option 3: Both (Full Setup)

Run Flask and Streamlit together:

```bash
# Terminal 1: Flask API
python app.py

# Terminal 2: Streamlit Dashboard
streamlit run dashboard/dashboard.py --server.port 8501
```

In the Streamlit dashboard, check "Use Flask API" to send requests through Flask.

---

## API Reference

### POST `/upload`

Upload an audio file for emotion analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `audio` (file, `.wav` or `.mp3`)

**Response (200):**

```json
{
  "status": "success",
  "duration": "02:15",
  "segments": [
    { "start": "00:00", "end": "00:04", "emotion": "neutral" },
    { "start": "00:04", "end": "00:08", "emotion": "happy" },
    { "start": "00:08", "end": "00:12", "emotion": "sad" }
  ],
  "raw_segments": [
    { "start": "00:00", "end": "00:02", "emotion": "neutral", "confidence": 0.8742 },
    { "start": "00:02", "end": "00:04", "emotion": "neutral", "confidence": 0.9103 }
  ],
  "summary": {
    "neutral": { "duration_seconds": 4.0, "percentage": 33.3 },
    "happy": { "duration_seconds": 4.0, "percentage": 33.3 },
    "sad": { "duration_seconds": 4.0, "percentage": 33.3 }
  }
}
```

### GET `/health`

Health check endpoint.

---

## Deployment

### Streamlit Cloud

1. Push the code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to: `dashboard/dashboard.py`
5. Deploy

**Note:** Streamlit Cloud has resource limits. The SpeechBrain model download may take a few minutes on first launch.

### Render

1. Push the code to a GitHub repository
2. Create a new Web Service on [render.com](https://render.com)
3. Connect your repository
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run dashboard/dashboard.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy

To run the Flask API on Render instead:
- **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
- Add `gunicorn` to `requirements.txt`

---

## Emotion Labels

The model detects four emotion categories:

| Emotion | Description |
|---|---|
| Neutral | Calm, flat tone |
| Happy | Positive, upbeat |
| Sad | Low energy, sorrowful |
| Angry | Aggressive, intense |

---

## Design Principles

- **Light theme only** with white background
- **Black, white, and gray** color palette exclusively
- **No gradients, no emojis, no decorative elements**
- Clean typography using Inter font family
- Professional, minimalistic interface

---

## License

This project is provided for educational and demonstration purposes.
# voice-emotion-analysis
