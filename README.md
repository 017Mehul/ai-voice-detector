
## 🌐 Live Demo

[🚀 Open VoxTrust App](https://ai-voice-detector-production-b5f3.up.railway.app/app)

🎙️ VoxTrust — AI Voice Detection Engine
> Detect deepfake voices in seconds using machine learning.
> Built for forensic analysis, security systems, and real-world deployment.
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
---
🧠 How It Works
Audio file is uploaded (MP3, WAV, or FLAC)
Audio is decoded, normalized, and validated using librosa
MFCC, spectral, and temporal features are extracted from the waveform
Features are passed into a trained Random Forest classifier
Model outputs a classification (`AI_GENERATED` or `HUMAN`) + confidence score
Result is stored in scan history and a PDF forensic report is generated
---
✨ Features
🔍 Detects AI-generated vs human voice with confidence scoring
🎵 Supports MP3, WAV, and FLAC audio formats
🖥️ Full forensic-style 7-screen web UI — no frontend build step needed
📄 REST API with scan history and downloadable PDF reports
📊 MFCC-based feature extraction + Random Forest classification
🛡️ Rate limiting, request validation, and duplicate detection
💾 Persistent scan history — no database required
---
📈 Model Performance
The repository contains a serialized classifier at `models/voice_classifier.joblib` and a calibration artifact at `models/calibration_params.joblib`. Performance figures should only be published alongside the exact dataset split, evaluation script, and model artifact used to produce them. The historical accuracy/precision/recall figures are omitted here because the checked-in training code describes a different model configuration (XGBoost with 78 features), so those figures cannot currently be independently reproduced from the repository as-is.

Before making benchmark claims, retrain/evaluate the deployed artifact using a held-out, speaker-disjoint test set and record the dataset version, class balance, confusion matrix, precision, recall, F1, and accuracy.
---
💼 Use Cases
🔐 Detect AI-generated scam calls and voice fraud
⚖️ Verify voice authenticity in legal and forensic investigations
📱 Content moderation for audio platforms
🏦 Financial institution voice authentication
🎙️ Deepfake detection for journalists and fact-checkers
---
🖥️ Screenshots
> Web UI available at `http://localhost:8000/app`
Screen	Description
Home	Landing page with feature overview
Workspace	Upload audio and configure analysis
Processing	Real-time analysis progress with execution stack
Report	Forensic verdict, confidence score, and model insights
Repository	Full scan history with filters and pagination
Diagnostics	Error details and retry options
Status	Live system health and engine status
---
🚀 Getting Started
Prerequisites
Python 3.10+
FFmpeg installed and on your `PATH` → Download FFmpeg
1. Clone the repository
```bash
git clone https://github.com/017Mehul/ai-voice-detector.git
cd ai-voice-detector
```
2. Create a virtual environment
```bash
python -m venv venv
```
Activate it:
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```
4. Set up environment
```bash
cp .env.example .env
```
Edit `.env` if needed — defaults work out of the box for local development.
5. Run the app
```bash
python -m voice_detector.main
```
Open your browser at http://localhost:8000/app
---
🌐 URLs
URL	Description
`http://localhost:8000/app`	Web UI
`http://localhost:8000/health`	Health check
`http://localhost:8000/health/detailed`	Detailed system status
`http://localhost:8000/api/v1/detect-voice`	Detection API (POST)
`http://localhost:8000/api/v1/history`	Scan history (GET)
`http://localhost:8000/api/v1/status`	Engine status (GET)
---
📡 API Reference
`POST /api/v1/detect-voice`
Request body:
```json
{
  "audio_base64": "<base64-encoded-audio>",
  "filename": "sample.wav",
  "file_format": "WAV",
  "duration_seconds": 4.2,
  "sample_rate_hz": 16000,
  "channels": "Mono"
}
```
Only `audio_base64` is required.
Response:
```json
{
  "classification": "AI_GENERATED",
  "confidence_score": 0.87
}
```
Response headers:
Header	Description
`X-Scan-ID`	Unique ID for this scan
`X-Report-URL`	PDF report download URL
`X-Request-ID`	Request trace ID
`X-Process-Time`	Processing time in seconds
Example with curl:
```bash
curl -X POST "http://localhost:8000/api/v1/detect-voice" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\":\"$(base64 -w0 sample.wav)\",\"filename\":\"sample.wav\",\"file_format\":\"WAV\"}"
```
---
⚙️ Configuration
Copy `.env.example` to `.env` and adjust as needed:
Variable	Default	Description
`API_HOST`	`0.0.0.0`	Server bind host
`API_PORT`	`8000`	Server port
`DEBUG`	`false`	Enable debug mode + API docs
`MODEL_PATH`	`./models/voice_classifier.joblib`	Classifier model path
`MAX_AUDIO_SIZE_MB`	`10`	Max upload size
`MIN_AUDIO_DURATION_SECONDS`	`2`	Minimum audio length
`MAX_AUDIO_DURATION_SECONDS`	`30`	Maximum audio length
`RATE_LIMIT_ENABLED`	`true`	Enable rate limiting
---
📁 Project Structure
```
voice_detector/
├── api/          # FastAPI routes, validation, middleware
├── audio/        # Audio decoding and normalization
├── features/     # MFCC and spectral feature extraction
├── models/       # Classifier, calibration, data models
├── services/     # Detection pipeline orchestration
├── utils/        # Config, logging, reporting, metrics
└── webui/        # Frontend HTML/JS/CSS (no build step)

tests/            # Pytest test suite
scripts/          # Setup and utility scripts
docs/             # Additional documentation
```
---
🧪 Running Tests
```bash
python -m pytest -q
```
---
🔁 Retraining the Model
If you have your own dataset in `dataset/Fake/` and `dataset/Real/`:
```bash
python train_model.py
```
---
📄 License
MIT — free to use, modify, and distribute.
---
👨‍💻 Author
Mehul Gupta
Aspiring Software Engineer | AI/ML Enthusiast | App and Web Developer
![GitHub](https://img.shields.io/badge/GitHub-017Mehul-181717?style=flat-square&logo=github)
