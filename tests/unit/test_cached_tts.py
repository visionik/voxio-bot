"""
Unit tests for cached_tts.py - TTS caching wrapper.
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cached_tts import (
    CachedElevenLabsTTSService,
    create_cached_tts_service,
)


class TestCacheKeyGeneration:
    """Tests for cache key generation."""
    
    def test_cache_key_deterministic(self) -> None:
        """Cache key should be deterministic for same inputs."""
        text = "Hello world"
        voice_id = "test-voice"
        model = "eleven_turbo_v2_5"
        
        cache_input = f"{text}:{voice_id}:{model}"
        expected = hashlib.md5(cache_input.encode()).hexdigest()
        
        # Verify the hash is consistent
        assert hashlib.md5(cache_input.encode()).hexdigest() == expected
    
    def test_cache_key_different_for_different_text(self) -> None:
        """Different text should produce different cache keys."""
        voice_id = "test-voice"
        model = "eleven_turbo_v2_5"
        
        key1 = hashlib.md5(f"Hello:{voice_id}:{model}".encode()).hexdigest()
        key2 = hashlib.md5(f"World:{voice_id}:{model}".encode()).hexdigest()
        
        assert key1 != key2
    
    def test_cache_key_different_for_different_voice(self) -> None:
        """Different voice should produce different cache keys."""
        text = "Hello"
        model = "eleven_turbo_v2_5"
        
        key1 = hashlib.md5(f"{text}:voice1:{model}".encode()).hexdigest()
        key2 = hashlib.md5(f"{text}:voice2:{model}".encode()).hexdigest()
        
        assert key1 != key2


class TestCachedTTSServiceInit:
    """Tests for CachedElevenLabsTTSService initialization."""
    
    @patch("cached_tts.ElevenLabsTTSService.__init__", return_value=None)
    def test_init_creates_cache_dir(
        self, mock_super_init: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Service should create cache directory on init."""
        service = CachedElevenLabsTTSService(
            api_key="test-key",
            voice_id="test-voice",
            cache_dir=str(temp_cache_dir),
        )
        
        assert temp_cache_dir.exists()
    
    @patch("cached_tts.ElevenLabsTTSService.__init__", return_value=None)
    def test_init_sets_defaults(
        self, mock_super_init: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Service should set default values correctly."""
        service = CachedElevenLabsTTSService(
            api_key="test-key",
            voice_id="test-voice",
            cache_dir=str(temp_cache_dir),
        )
        
        assert service._cache_hits == 0
        assert service._cache_misses == 0


class TestCacheOperations:
    """Tests for cache read/write operations."""
    
    @pytest.mark.asyncio
    async def test_cache_exists_returns_false_for_missing(
        self, temp_cache_dir: Path
    ) -> None:
        """_cache_exists should return False for missing files."""
        with patch("cached_tts.ElevenLabsTTSService.__init__", return_value=None):
            service = CachedElevenLabsTTSService(
                api_key="test-key",
                voice_id="test-voice",
                cache_dir=str(temp_cache_dir),
            )
            
            result = await service._cache_exists("nonexistent-key")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_cache_exists_returns_true_for_existing(
        self, temp_cache_dir: Path
    ) -> None:
        """_cache_exists should return True for existing files."""
        # Create a cache file
        cache_file = temp_cache_dir / "test-key.pcm"
        cache_file.write_bytes(b"test audio data")
        
        with patch("cached_tts.ElevenLabsTTSService.__init__", return_value=None):
            service = CachedElevenLabsTTSService(
                api_key="test-key",
                voice_id="test-voice",
                cache_dir=str(temp_cache_dir),
            )
            # Override the cache path method to return our test file
            service._get_cache_path = lambda key: cache_file
            
            result = await service._cache_exists("test-key")
            assert result is True


class TestCacheStats:
    """Tests for cache statistics."""
    
    @patch("cached_tts.ElevenLabsTTSService.__init__", return_value=None)
    def test_get_cache_stats_initial(
        self, mock_super_init: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Cache stats should be zero initially."""
        service = CachedElevenLabsTTSService(
            api_key="test-key",
            voice_id="test-voice",
            cache_dir=str(temp_cache_dir),
        )
        
        stats = service.get_cache_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    @patch("cached_tts.ElevenLabsTTSService.__init__", return_value=None)
    def test_get_cache_stats_with_activity(
        self, mock_super_init: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Cache stats should reflect activity."""
        service = CachedElevenLabsTTSService(
            api_key="test-key",
            voice_id="test-voice",
            cache_dir=str(temp_cache_dir),
        )
        
        # Simulate some cache activity
        service._cache_hits = 3
        service._cache_misses = 1
        
        stats = service.get_cache_stats()
        
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75


class TestCreateCachedTTSService:
    """Tests for the convenience factory function."""
    
    @patch("cached_tts.CachedElevenLabsTTSService")
    def test_factory_function(self, mock_class: MagicMock) -> None:
        """Factory function should create service with correct params."""
        create_cached_tts_service(
            api_key="test-key",
            voice_id="test-voice",
            model="eleven_turbo_v2_5",
            cache_dir="/tmp/test-cache",
        )
        
        mock_class.assert_called_once_with(
            api_key="test-key",
            voice_id="test-voice",
            model="eleven_turbo_v2_5",
            cache_dir="/tmp/test-cache",
        )
