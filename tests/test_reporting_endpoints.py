"""Integration coverage for scan history and PDF reporting endpoints."""

from __future__ import annotations

from io import BytesIO
import base64
import math
import wave
from unittest.mock import patch

from fastapi.testclient import TestClient

from voice_detector.api.app import create_app
from voice_detector.models.data_models import DetectionResponse


def _generate_wav_base64(duration_seconds: float = 2.2, sample_rate: int = 16000) -> str:
    """Create a small valid WAV payload for API integration tests."""
    frames = int(duration_seconds * sample_rate)
    buffer = BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        waveform = bytearray()
        for index in range(frames):
            sample = int(16000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            waveform.extend(sample.to_bytes(2, byteorder="little", signed=True))

        wav_file.writeframes(bytes(waveform))

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_detection_history_and_pdf_report_round_trip(tmp_path, monkeypatch):
    history_path = tmp_path / "scan_history.json"
    monkeypatch.setenv("VOICE_DETECTOR_HISTORY_PATH", str(history_path))

    payload = {
        "audio_base64": _generate_wav_base64(),
        "filename": "evidence_case_alpha.wav",
        "file_format": "WAV",
        "duration_seconds": 2.2,
        "sample_rate_hz": 16000,
        "channels": "Mono",
        "bitrate_kbps": "256 kbps",
    }

    class DummyVoiceDetectionService:
        def detect_voice(self, detection_request, request_id):
            return DetectionResponse(classification="HUMAN", confidence_score=0.91), None

    class DummyServiceManager:
        def is_initialized(self):
            return True

        def get_service(self, service_name):
            assert service_name == "voice_detection_service"
            return DummyVoiceDetectionService()

    with patch("voice_detector.services.get_service_manager", return_value=DummyServiceManager()):
        client = TestClient(create_app())
        response = client.post("/api/v1/detect-voice", json=payload)
        assert response.status_code == 200

        scan_id = response.headers["X-Scan-ID"]
        report_url = response.headers["X-Report-URL"]
        assert report_url.endswith(f"/{scan_id}.pdf")

        history_response = client.get("/api/v1/history")
        assert history_response.status_code == 200
        history_payload = history_response.json()
        assert history_payload["count"] >= 1
        first_item = history_payload["items"][0]
        assert first_item["id"] == scan_id
        assert first_item["name"] == payload["filename"]
        assert first_item["format"] == payload["file_format"]

        detail_response = client.get(f"/api/v1/history/{scan_id}")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["id"] == scan_id
        assert detail_payload["summary"]

        pdf_response = client.get(report_url)
        assert pdf_response.status_code == 200
        assert pdf_response.headers["content-type"] == "application/pdf"
        assert pdf_response.content.startswith(b"%PDF-")
