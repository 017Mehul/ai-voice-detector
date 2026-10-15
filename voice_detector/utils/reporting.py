"""History persistence and PDF report generation utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import hashlib
import json
import os
import secrets


@dataclass
class ScanRecord:
    """Persisted metadata for a completed voice detection scan."""

    scan_id: str
    request_id: str
    timestamp: str
    classification: str
    confidence_score: float
    filename: str
    file_format: str
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    channels: Optional[str] = None
    bitrate_kbps: Optional[str] = None
    fingerprint: Optional[str] = None
    summary: Optional[str] = None
    waveform_bars: Optional[List[float]] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a frontend-friendly dictionary."""
        waveform_tint = "wave-gold" if self.classification == "AI_GENERATED" else "wave-cyan"
        if self.classification not in {"AI_GENERATED", "HUMAN"}:
            waveform_tint = "wave-neutral"

        return {
            "id": self.scan_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "classification": self.classification,
            "confidence": float(self.confidence_score),
            "confidence_score": float(self.confidence_score),
            "name": self.filename,
            "format": self.file_format,
            "duration": self.duration_seconds,
            "sampleRate": self.sample_rate_hz,
            "sampleRateLabel": f"{self.sample_rate_hz / 1000:.1f} kHz" if self.sample_rate_hz else None,
            "channels": self.channels,
            "bitrate": self.bitrate_kbps,
            "fingerprint": self.fingerprint,
            "summary": self.summary,
            "waveformTint": waveform_tint,
            "reportUrl": f"/api/v1/reports/{self.scan_id}.pdf",
            "waveform": self.waveform_bars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScanRecord":
        """Create a scan record from persisted JSON."""
        if "scan_id" in data:
            return cls(**data)

        return cls(
            scan_id=data.get("id", create_scan_id()),
            request_id=data.get("request_id", "unknown"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            classification=data.get("classification", "INCONCLUSIVE"),
            confidence_score=float(data.get("confidence_score", data.get("confidence", 0.0))),
            filename=data.get("filename", data.get("name", "UNKNOWN_AUDIO")),
            file_format=data.get("file_format", data.get("format", "WAV")),
            duration_seconds=data.get("duration_seconds", data.get("duration")),
            sample_rate_hz=data.get("sample_rate_hz", data.get("sampleRate")),
            channels=data.get("channels"),
            bitrate_kbps=data.get("bitrate_kbps", data.get("bitrate")),
            fingerprint=data.get("fingerprint"),
            summary=data.get("summary"),
            waveform_bars=data.get("waveform_bars", data.get("waveform")),
        )


def create_scan_id() -> str:
    """Create a short stable scan identifier."""
    return f"VX-{secrets.token_hex(4).upper()}"


def create_audio_fingerprint(audio_base64: str) -> str:
    """Create a compact forensic fingerprint from the raw audio payload."""
    digest = hashlib.sha256(audio_base64.encode("utf-8")).digest()[:16]
    return ":".join(f"{byte:02X}" for byte in digest)


def default_summary(classification: str, confidence_score: float) -> str:
    """Create a human-readable summary for history records and PDF output."""
    percent = round(confidence_score * 100)

    if classification == "AI_GENERATED":
        return (
            f"The evidence shows synthetic acoustic regularity consistent with AI-generated speech. "
            f"The final confidence score is {percent}%."
        )

    if classification == "HUMAN":
        return (
            f"The evidence retains natural variability consistent with live human speech. "
            f"The final confidence score is {percent}%."
        )

    return (
        f"The evidence contains mixed indicators and remains inconclusive. "
        f"The final confidence score is {percent}%."
    )


class ScanHistoryStore:
    """Persist completed scan metadata to a JSON file in the repository data directory."""

    def __init__(self, path: Optional[Path] = None, max_entries: int = 250):
        base_path = path or self._default_path()
        self.path = Path(base_path)
        self.max_entries = max_entries
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_path() -> Path:
        override = os.getenv("VOICE_DETECTOR_HISTORY_PATH")
        if override:
            return Path(override)

        return Path(__file__).resolve().parents[2] / "data" / "scan_history.json"

    def add_record(self, record: ScanRecord) -> ScanRecord:
        """Insert a record at the top of the history file."""
        with self._lock:
            records = self._load_unlocked()
            records = [entry for entry in records if entry.scan_id != record.scan_id]
            records.insert(0, record)
            self._save_unlocked(records[: self.max_entries])
        return record

    def list_records(self, limit: int = 50, session_id: Optional[str] = None) -> List[ScanRecord]:
        """Return only records owned by the supplied browser session."""
        if not session_id:
            return []
        with self._lock:
            owned = [record for record in self._load_unlocked() if record.session_id == session_id]
            return owned[:limit]

    def get_record(self, scan_id: str, session_id: Optional[str] = None) -> Optional[ScanRecord]:
        """Return a record only when it belongs to the supplied browser session."""
        if not session_id:
            return None
        with self._lock:
            for record in self._load_unlocked():
                if record.scan_id == scan_id and record.session_id == session_id:
                    return record
        return None

    def _load_unlocked(self) -> List[ScanRecord]:
        if not self.path.exists():
            return []

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(raw, list):
            return []

        records = [ScanRecord.from_dict(item) for item in raw if isinstance(item, dict)]
        return sorted(records, key=lambda item: item.timestamp, reverse=True)

    def _save_unlocked(self, records: List[ScanRecord]) -> None:
        payload = [asdict(record) for record in records]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_history_store: Optional[ScanHistoryStore] = None
_history_store_path: Optional[Path] = None


def get_scan_history_store() -> ScanHistoryStore:
    """Return a cached history store that tracks the current configured path."""
    global _history_store, _history_store_path

    resolved_path = ScanHistoryStore._default_path()
    if _history_store is None or _history_store_path != resolved_path:
        _history_store = ScanHistoryStore(resolved_path)
        _history_store_path = resolved_path

    return _history_store


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_report_pdf(record: ScanRecord) -> bytes:
    """Generate a simple single-page PDF report without third-party dependencies."""
    lines = [
        ("VOXTRUST FORENSIC REPORT", 18),
        ("", 12),
        (f"Scan ID: {record.scan_id}", 12),
        (f"Generated: {record.timestamp}", 12),
        (f"File: {record.filename}", 12),
        (f"Format: {record.file_format}", 12),
        (
            f"Verdict: {record.classification.replace('_', ' ')} "
            f"({round(record.confidence_score * 100)}%)",
            12,
        ),
        (f"Sample Rate: {record.sample_rate_hz or 'Unknown'}", 12),
        (f"Channels: {record.channels or 'Unknown'}", 12),
        (f"Bitrate: {record.bitrate_kbps or 'Unknown'}", 12),
        (f"Fingerprint: {record.fingerprint or 'Unavailable'}", 12),
        ("", 12),
        ("Summary", 14),
        (record.summary or default_summary(record.classification, record.confidence_score), 12),
    ]

    content_commands: List[str] = []
    y_position = 760
    for text, font_size in lines:
        if text:
            content_commands.append(
                f"BT /F1 {font_size} Tf 1 0 0 1 50 {y_position} Tm "
                f"({_escape_pdf_text(text)}) Tj ET"
            )
        y_position -= 22 if font_size >= 14 else 18

    content = "\n".join(content_commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)
