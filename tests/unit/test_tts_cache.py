"""
Unit tests for tts_cache.py - Unified TTS cache implementation.
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tts_cache import (
    CacheConfig,
    TTSCache,
    ALL_FILLERS,
)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""
    
    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = CacheConfig()
        
        assert config.cache_dir == "~/.cache/voxio-tts"
        assert config.max_age_days == 30
        assert config.sample_rate == 24000
        assert config.model == "eleven_turbo_v2_5"
    
    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = CacheConfig(
            cache_dir="/custom/cache",
            max_age_days=7,
            sample_rate=44100,
            model="eleven_multilingual_v2",
        )
        
        assert config.cache_dir == "/custom/cache"
        assert config.max_age_days == 7
        assert config.sample_rate == 44100
        assert config.model == "eleven_multilingual_v2"
    
    def test_dir_property_expands_path(self) -> None:
        """dir property should expand ~ to home directory."""
        config = CacheConfig(cache_dir="~/.cache/test")
        
        # Call __post_init__ which sets up the path
        assert "~" not in str(config.dir)


class TestFillerPhrases:
    """Tests for filler phrase constants."""
    
    def test_filler_phrases_exist(self) -> None:
        """Filler phrases should be defined."""
        assert len(ALL_FILLERS) > 0
    
    def test_filler_phrases_are_strings(self) -> None:
        """All filler phrases should be strings."""
        for phrase in ALL_FILLERS:
            assert isinstance(phrase, str)
            assert len(phrase) > 0
    
    def test_common_phrases_included(self) -> None:
        """Common acknowledgment phrases should be included."""
        common = ["Hmm.", "Got it.", "Okay."]
        for phrase in common:
            assert phrase in ALL_FILLERS, f"Missing: {phrase}"


class TestTTSCache:
    """Tests for TTSCache class."""
    
    def test_init_creates_cache_dir(self, temp_cache_dir: Path) -> None:
        """Cache should create directory on init."""
        config = CacheConfig(cache_dir=str(temp_cache_dir / "new_cache"))
        cache = TTSCache(config)
        
        assert config.dir.exists()
    
    def test_cache_key_generation(self, temp_cache_dir: Path) -> None:
        """Cache keys should be deterministic."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        key1 = cache._get_key("Hello", "voice1")
        key2 = cache._get_key("Hello", "voice1")
        key3 = cache._get_key("World", "voice1")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_cache_path_structure(self, temp_cache_dir: Path) -> None:
        """Cache paths should follow expected structure."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        key = cache._get_key("test text", "voice123")
        path = cache._get_path(key, "voice123")
        
        assert path.parent.name == "voice123"
        assert path.suffix == ".pcm"
    
    @pytest.mark.asyncio
    async def test_get_returns_none_when_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """get() should return None for uncached items."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        result = await cache.get("Missing text", "voice1")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_put_and_get(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """Should be able to put and get audio data."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        # Put
        await cache.put("Test text", "voice1", sample_audio_bytes)
        
        # Get
        result = await cache.get("Test text", "voice1")
        
        assert result == sample_audio_bytes
    
    @pytest.mark.asyncio
    async def test_exists_returns_false_when_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """exists() should return False for uncached items."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        result = await cache.exists("Missing text", "voice1")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_returns_true_when_cached(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """exists() should return True for cached items."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        await cache.put("Test text", "voice1", sample_audio_bytes)
        result = await cache.exists("Test text", "voice1")
        
        assert result is True


class TestTTSCacheStats:
    """Tests for TTSCache statistics."""
    
    def test_initial_stats(self, temp_cache_dir: Path) -> None:
        """Stats should be zero initially."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        stats = cache.stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_stats_track_hits(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """Stats should track cache hits."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        # Put and get (hit)
        await cache.put("Test", "voice1", sample_audio_bytes)
        await cache.get("Test", "voice1")
        
        stats = cache.stats()
        
        assert stats["hits"] == 1
    
    @pytest.mark.asyncio
    async def test_stats_misses_start_at_zero(self, temp_cache_dir: Path) -> None:
        """Stats misses should start at zero (tracked by CachedTTSService)."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        # Note: TTSCache.get() doesn't track misses directly
        # Misses are tracked in CachedTTSService.run_tts()
        await cache.get("Missing", "voice1")
        
        stats = cache.stats()
        
        # Misses remain 0 at TTSCache level
        assert stats["misses"] == 0


class TestTTSCacheCleanup:
    """Tests for TTSCache cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_clear_removes_all_files(
        self, temp_cache_dir: Path, sample_audio_bytes: bytes
    ) -> None:
        """clear() should remove cached files."""
        config = CacheConfig(cache_dir=str(temp_cache_dir))
        cache = TTSCache(config)
        
        # Add some files
        await cache.put("Test 1", "voice1", sample_audio_bytes)
        await cache.put("Test 2", "voice1", sample_audio_bytes)
        
        # Verify files exist
        voice_dir = config.dir / "voice1"
        assert len(list(voice_dir.glob("*.pcm"))) == 2
        
        # Clear
        removed = await cache.clear("voice1")
        
        # Verify files removed
        assert removed == 2
        assert len(list(voice_dir.glob("*.pcm"))) == 0
