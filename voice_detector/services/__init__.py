"""
Services Layer - Dependency Injection and Component Wiring

This module provides service management and dependency injection for the
AI Voice Detection System components.
"""

from .service_manager import ServiceManager, get_service_manager, initialize_services, shutdown_services
from .voice_detection_service import VoiceDetectionService

__all__ = [
    'ServiceManager',
    'get_service_manager',
    'initialize_services',
    'shutdown_services',
    'VoiceDetectionService'
]