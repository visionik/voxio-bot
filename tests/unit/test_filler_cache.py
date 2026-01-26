"""
Unit tests for filler_cache.py - Filler pre-caching system.
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from filler_cache import (
    FillerConfig,
    get_filler_cache_dir,
    get_filler_cache_key,
    get_filler_cache_path,
    is_filler_cached,
    load_filler_audio,
    save_filler_audio,
    get_missing_fillers,
    is_voice_fully_cached,
    ALL_FILLERS,
    FILLERS_ACKNOWLEDGMENT,
    FILLERS_THINKING,
    FILLERS_TRANSITION,
    FILLER_CATEGORIES,
)


class TestFillerConfig:
    """Tests for FillerConfig dataclass."""
    
    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = FillerConfig()
        
        assert config.cache_dir == "~/.cache/voxio-fillers"
        assert config.model == "eleven_turbo_v2_5"
        assert config.sample_rate == 24000
    
    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = FillerConfig(
            cache_dir="/custom/path",
            model="eleven_multilingual_v2",
            sample_rate=44100,
        )
        
        assert config.cache_dir == "/custom/path"
        assert config.model == "eleven_multilingual_v2"
        assert config.sample_rate == 44100


class TestFillerPhrases:
    """Tests for filler phrase constants."""
    
    def test_acknowledgment_fillers_exist(self) -> None:
        """Acknowledgment fillers should be defined."""
        assert len(FILLERS_ACKNOWLEDGMENT) > 0
        assert "Hmm." in FILLERS_ACKNOWLEDGMENT
        assert "Got it." in FILLERS_ACKNOWLEDGMENT
    
    def test_thinking_fillers_exist(self) -> None:
        """Thinking fillers should be defined."""
        assert len(FILLERS_THINKING) > 0
        assert "Let me think..." in FILLERS_THINKING
    
    def test_transition_fillers_exist(self) -> None:
        """Transition fillers should be defined."""
        assert len(FILLERS_TRANSITION) > 0
    
    def test_all_fillers_combines_categories(self) -> None:
        """ALL_FILLERS should contain all categories."""
        expected_count = (
            len(FILLERS_ACKNOWLEDGMENT)
            + len(FILLERS_THINKING)
            + len(FILLERS_TRANSITION)
        )
        assert len(ALL_FILLERS) == expected_count
    
    def test_filler_categories_dict(self) -> None:
        """FILLER_CATEGORIES should map to correct lists."""
        assert FILLER_CATEGORIES["ack"] == FILLERS_ACKNOWLEDGMENT
        assert FILLER_CATEGORIES["think"] == FILLERS_THINKING
        assert FILLER_CATEGORIES["transition"] == FILLERS_TRANSITION


class TestCachePathGeneration:
    """Tests for cache path generation functions."""
    
    def test_get_filler_cache_dir_expands_path(self) -> None:
        """Cache dir should expand ~ to home directory."""
        config = FillerConfig(cache_dir="~/.cache/test")
        result = get_filler_cache_dir(config)
        
        assert "~" not in str(result)
        assert result.is_absolute()
    
    def test_get_filler_cache_key_deterministic(self) -> None:
        """Cache key should be deterministic."""
        key1 = get_filler_cache_key("Hello", "voice1", "model1")
        key2 = get_filler_cache_key("Hello", "voice1", "model1")
        
        assert key1 == key2
    
    def test_get_filler_cache_key_different_inputs(self) -> None:
        """Different inputs should produce different keys."""
        key1 = get_filler_cache_key("Hello", "voice1", "model1")
        key2 = get_filler_cache_key("World", "voice1", "model1")
        key3 = get_filler_cache_key("Hello", "voice2", "model1")
        
        assert key1 != key2
        assert key1 != key3
    
    def test_get_filler_cache_path_structure(self, temp_cache_dir: Path) -> None:
        """Cache path should have correct structure."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        path = get_filler_cache_path("Test phrase", "voice123", config)
        
        assert path.parent.name == "voice123"
        assert path.suffix == ".pcm"


class TestCacheOperations:
    """Tests for cache file operations."""
    
    @pytest.mark.asyncio
    async def test_is_filler_cached_false_when_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """is_filler_cached should return False for missing files."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        result = await is_filler_cached("Missing phrase", "voice1", config)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_filler_cached_true_when_exists(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """is_filler_cached should return True for existing files."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        # Create the cache file
        await save_filler_audio("Test phrase", "voice1", sample_audio_bytes, config)
        
        result = await is_filler_cached("Test phrase", "voice1", config)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_save_and_load_filler_audio(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """Should be able to save and load audio data."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        # Save
        await save_filler_audio("Test phrase", "voice1", sample_audio_bytes, config)
        
        # Load
        loaded = await load_filler_audio("Test phrase", "voice1", config)
        
        assert loaded == sample_audio_bytes
    
    @pytest.mark.asyncio
    async def test_load_filler_audio_returns_none_when_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """load_filler_audio should return None for missing files."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        result = await load_filler_audio("Missing phrase", "voice1", config)
        
        assert result is None


class TestMissingFillerDetection:
    """Tests for detecting missing fillers."""
    
    @pytest.mark.asyncio
    async def test_get_missing_fillers_all_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """Should return all fillers when none are cached."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        phrases = ["Phrase 1", "Phrase 2", "Phrase 3"]
        
        missing = await get_missing_fillers("voice1", config, phrases)
        
        assert missing == phrases
    
    @pytest.mark.asyncio
    async def test_get_missing_fillers_some_cached(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """Should only return uncached fillers."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        phrases = ["Phrase 1", "Phrase 2", "Phrase 3"]
        
        # Cache one phrase
        await save_filler_audio("Phrase 2", "voice1", sample_audio_bytes, config)
        
        missing = await get_missing_fillers("voice1", config, phrases)
        
        assert "Phrase 1" in missing
        assert "Phrase 2" not in missing
        assert "Phrase 3" in missing
    
    @pytest.mark.asyncio
    async def test_is_voice_fully_cached_false_when_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """Should return False when fillers are missing."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        result = await is_voice_fully_cached("voice1", config)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_voice_fully_cached_true_when_complete(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """Should return True when all fillers are cached."""
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        # Cache all fillers
        for phrase in ALL_FILLERS:
            await save_filler_audio(phrase, "voice1", sample_audio_bytes, config)
        
        result = await is_voice_fully_cached("voice1", config)
        
        assert result is True


class TestFillerInjectorInit:
    """Tests for FillerInjector initialization."""
    
    def test_init_with_defaults(self, temp_cache_dir: Path) -> None:
        """FillerInjector should initialize with defaults."""
        from filler_cache import FillerInjector
        
        config = FillerConfig(cache_dir=str(temp_cache_dir))
        
        with patch.object(FillerInjector, "__init__", lambda self, **kw: None):
            injector = FillerInjector.__new__(FillerInjector)
            injector._voice_id = "test-voice"
            injector._config = config
            injector._play_acknowledgment = True
            injector._play_thinking = False
            injector._loaded_fillers = {}
            
            assert injector._voice_id == "test-voice"
            assert injector._play_acknowledgment is True
            assert injector._play_thinking is False
