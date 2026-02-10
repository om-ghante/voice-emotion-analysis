"""
Flask API Layer for Voice Emotion Analysis.
Accepts audio file uploads and returns structured emotion analysis results.
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from services.emotion_service import analyze_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_audio():
    """
    Handle audio file upload and emotion analysis.

    Accepts: POST with a file field named 'audio'.
    Returns: JSON with duration, segments, raw_segments, and summary.
    """
    # Validate file presence
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use the 'audio' field."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    logger.info(f"File uploaded: {filename} -> {filepath}")

    try:
        # Run emotion analysis
        result = analyze_audio(filepath, chunk_duration=2.0)
        logger.info(f"Analysis complete: {len(result['segments'])} segments detected")

        return jsonify({
            "status": "success",
            "duration": result["duration"],
            "segments": result["segments"],
            "raw_segments": result["raw_segments"],
            "summary": result["summary"],
        }), 200

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up uploaded file: {filepath}")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
