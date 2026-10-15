# Production Docker image for VoxTrust
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Both packages are currently used by the application and regression tests.
COPY voice_detector/ ./voice_detector/
COPY ai_voice_detector/ ./ai_voice_detector/
COPY models/ ./models/
COPY setup.py README.md ./

RUN pip install --no-cache-dir -e . \
    && test -s models/voice_classifier.joblib \
    && test -s models/calibration_params.joblib

RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "voice_detector.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
