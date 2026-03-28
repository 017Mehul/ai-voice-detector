"""Compatibility package for the legacy ``ai_voice_detector`` import path.

The repository's canonical package is now ``voice_detector``. A large portion of
the tests, deployment files, and serialized model artifacts still reference the
old package name, so we alias the old module path to the new implementation.
"""

from __future__ import annotations

import importlib
import sys

import voice_detector as _voice_detector

__version__ = _voice_detector.__version__
__author__ = getattr(_voice_detector, "__author__", "AI Voice Detection Team")


def _alias_module(legacy_name: str, current_name: str):
    module = importlib.import_module(current_name)
    sys.modules[legacy_name] = module
    return module


_ALIASES = [
    ("ai_voice_detector.utils.logging_config", "voice_detector.utils.logging_config"),
    ("ai_voice_detector.utils.config", "voice_detector.utils.config"),
    ("ai_voice_detector.utils.helpers", "voice_detector.utils.helpers"),
    ("ai_voice_detector.utils.performance", "voice_detector.utils.performance"),
    ("ai_voice_detector.utils.prometheus_metrics", "voice_detector.utils.prometheus_metrics"),
    ("ai_voice_detector.utils.reporting", "voice_detector.utils.reporting"),
    ("ai_voice_detector.utils.synthetic_dataset", "voice_detector.utils.synthetic_dataset"),
    ("ai_voice_detector.utils", "voice_detector.utils"),
    ("ai_voice_detector.models.data_models", "voice_detector.models.data_models"),
    ("ai_voice_detector.models.classifier", "voice_detector.models.classifier"),
    ("ai_voice_detector.models.calibration", "voice_detector.models.calibration"),
    ("ai_voice_detector.models.dataset_preparation", "voice_detector.models.dataset_preparation"),
    ("ai_voice_detector.models.evaluation", "voice_detector.models.evaluation"),
    ("ai_voice_detector.models.model_selection", "voice_detector.models.model_selection"),
    ("ai_voice_detector.models.training", "voice_detector.models.training"),
    ("ai_voice_detector.models", "voice_detector.models"),
    ("ai_voice_detector.audio.processor", "voice_detector.audio.processor"),
    ("ai_voice_detector.audio", "voice_detector.audio"),
    ("ai_voice_detector.features.extractor", "voice_detector.features.extractor"),
    ("ai_voice_detector.features", "voice_detector.features"),
    ("ai_voice_detector.services.voice_detection_service", "voice_detector.services.voice_detection_service"),
    ("ai_voice_detector.services.service_manager", "voice_detector.services.service_manager"),
    ("ai_voice_detector.services", "voice_detector.services"),
    ("ai_voice_detector.api.validation", "voice_detector.api.validation"),
    ("ai_voice_detector.api.responses", "voice_detector.api.responses"),
    ("ai_voice_detector.api.rate_limiting", "voice_detector.api.rate_limiting"),
    ("ai_voice_detector.api.endpoints", "voice_detector.api.endpoints"),
    ("ai_voice_detector.api.app", "voice_detector.api.app"),
    ("ai_voice_detector.api", "voice_detector.api"),
    ("ai_voice_detector.main", "voice_detector.main"),
]

for legacy_name, current_name in _ALIASES:
    module = _alias_module(legacy_name, current_name)
    short_name = legacy_name.removeprefix("ai_voice_detector.")
    if "." not in short_name:
        globals()[short_name] = module
