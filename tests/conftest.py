"""
Pytest configuration and shared fixtures for Voxio Bot tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


# =============================================================================
# Temporary Directories
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache_dir(temp_dir: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# =============================================================================
# Mock Audio Data
# =============================================================================

@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Generate sample PCM audio bytes for testing."""
    # 100ms of silence at 24kHz, 16-bit mono
    samples = 2400  # 24000 * 0.1
    return bytes(samples * 2)  # 2 bytes per sample


@pytest.fixture
def sample_audio_chunks(sample_audio_bytes: bytes) -> list[bytes]:
    """Generate sample audio chunks for testing."""
    chunk_size = 480  # 20ms at 24kHz
    return [
        sample_audio_bytes[i : i + chunk_size]
        for i in range(0, len(sample_audio_bytes), chunk_size)
    ]


# =============================================================================
# Mock ElevenLabs Client
# =============================================================================

@pytest.fixture
def mock_elevenlabs_client(sample_audio_bytes: bytes) -> MagicMock:
    """Create a mock ElevenLabs client."""
    client = MagicMock()
    
    # Mock text_to_speech.convert to return audio generator
    def mock_convert(*args, **kwargs):
        # Return generator that yields audio chunks
        chunk_size = 480
        for i in range(0, len(sample_audio_bytes), chunk_size):
            yield sample_audio_bytes[i : i + chunk_size]
    
    client.text_to_speech.convert = mock_convert
    return client


# =============================================================================
# Async Utilities
# =============================================================================

@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Filler Phrases
# =============================================================================

@pytest.fixture
def sample_filler_phrases() -> list[str]:
    """Sample filler phrases for testing."""
    return [
        "Hmm.",
        "Got it.",
        "Let me think...",
    ]


# =============================================================================
# Mock Frame Classes
# =============================================================================

@pytest.fixture
def mock_audio_frame(sample_audio_bytes: bytes) -> MagicMock:
    """Create a mock AudioRawFrame."""
    frame = MagicMock()
    frame.audio = sample_audio_bytes[:480]
    frame.sample_rate = 24000
    frame.num_channels = 1
    return frame


@pytest.fixture
def mock_tts_started_frame() -> MagicMock:
    """Create a mock TTSStartedFrame."""
    return MagicMock(name="TTSStartedFrame")


@pytest.fixture
def mock_tts_stopped_frame() -> MagicMock:
    """Create a mock TTSStoppedFrame."""
    return MagicMock(name="TTSStoppedFrame")
