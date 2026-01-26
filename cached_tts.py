"""
Caching wrapper for ElevenLabs TTS service.

Streams audio on first generation while saving to cache in background,
then serves directly from cache on subsequent identical requests.

Usage:
    from cached_tts import CachedElevenLabsTTSService
    
    tts = CachedElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="QzTKubutNn9TjrB7Xb2Q",
        model="eleven_turbo_v2_5",
        cache_dir="~/.cache/voxio-tts",
    )
"""

import os
import hashlib
import asyncio
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Optional, AsyncGenerator
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


class CachedElevenLabsTTSService(ElevenLabsTTSService):
    """
    ElevenLabs TTS service with transparent caching.
    
    On cache miss: Streams audio immediately while saving chunks to cache file.
    On cache hit: Streams audio directly from cache file.
    
    Cache key is MD5 hash of (text + voice_id + model).
    """
    
    def __init__(
        self,
        *,
        cache_dir: str = "~/.cache/voxio-tts",
        cache_max_age_days: int = 30,
        **kwargs,
    ):
        """
        Initialize cached TTS service.
        
        Args:
            cache_dir: Directory to store cached audio files
            cache_max_age_days: Max age of cache files before cleanup (0 = no cleanup)
            **kwargs: Passed to ElevenLabsTTSService
        """
        super().__init__(**kwargs)
        
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_max_age_days = cache_max_age_days
        self._voice_id = kwargs.get("voice_id", "")
        self._model = kwargs.get("model", "eleven_turbo_v2_5")
        
        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"🗄️ TTS cache initialized: {self._cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text + voice + model."""
        cache_input = f"{text}:{self._voice_id}:{self._model}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file for given key."""
        return self._cache_dir / f"{cache_key}.pcm"
    
    async def _cache_exists(self, cache_key: str) -> bool:
        """Check if cache file exists."""
        cache_path = self._get_cache_path(cache_key)
        try:
            return await aiofiles.os.path.exists(cache_path)
        except Exception:
            return False
    
    async def _read_from_cache(self, cache_key: str) -> AsyncGenerator[bytes, None]:
        """Read audio chunks from cache file."""
        cache_path = self._get_cache_path(cache_key)
        chunk_size = 4096  # Match typical audio frame size
        
        try:
            async with aiofiles.open(cache_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            raise
    
    async def _write_to_cache(self, cache_key: str, audio_chunks: list[bytes]):
        """Write audio chunks to cache file (background task)."""
        cache_path = self._get_cache_path(cache_key)
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                for chunk in audio_chunks:
                    await f.write(chunk)
            
            # Atomic rename
            await aiofiles.os.rename(temp_path, cache_path)
            logger.debug(f"💾 Cached TTS: {cache_key[:8]}... ({len(audio_chunks)} chunks)")
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            # Clean up temp file if it exists
            try:
                await aiofiles.os.remove(temp_path)
            except Exception:
                pass
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech from text, using cache when available.
        
        On cache hit: Stream directly from cache file.
        On cache miss: Stream from ElevenLabs while caching chunks.
        """
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if await self._cache_exists(cache_key):
            self._cache_hits += 1
            logger.debug(f"🎯 TTS cache hit: {cache_key[:8]}... (hits: {self._cache_hits})")
            
            # Yield start frame
            yield TTSStartedFrame()
            
            # Stream from cache
            async for chunk in self._read_from_cache(cache_key):
                yield AudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
            
            # Yield stop frame
            yield TTSStoppedFrame()
            return
        
        # Cache miss - generate and cache
        self._cache_misses += 1
        logger.debug(f"📝 TTS cache miss: {cache_key[:8]}... (misses: {self._cache_misses})")
        
        # Collect chunks for caching while streaming
        audio_chunks: list[bytes] = []
        
        async for frame in super().run_tts(text):
            # Capture audio frames for caching
            if isinstance(frame, AudioRawFrame):
                audio_chunks.append(frame.audio)
            
            # Stream immediately
            yield frame
        
        # Write to cache in background (don't block streaming)
        if audio_chunks:
            asyncio.create_task(self._write_to_cache(cache_key, audio_chunks))
    
    async def cleanup_old_cache(self):
        """Remove cache files older than max_age_days."""
        if self._cache_max_age_days <= 0:
            return
        
        import time
        max_age_secs = self._cache_max_age_days * 24 * 60 * 60
        now = time.time()
        removed = 0
        
        try:
            for cache_file in self._cache_dir.glob("*.pcm"):
                stat = await aiofiles.os.stat(cache_file)
                if now - stat.st_mtime > max_age_secs:
                    await aiofiles.os.remove(cache_file)
                    removed += 1
            
            if removed:
                logger.info(f"🧹 Cleaned up {removed} old TTS cache files")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "cache_dir": str(self._cache_dir),
        }
    
    async def clear_cache(self):
        """Clear all cached audio files."""
        removed = 0
        try:
            for cache_file in self._cache_dir.glob("*.pcm"):
                await aiofiles.os.remove(cache_file)
                removed += 1
            logger.info(f"🗑️ Cleared {removed} TTS cache files")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")


# Convenience function for integration with existing code
def create_cached_tts_service(
    api_key: str,
    voice_id: str,
    model: str = "eleven_turbo_v2_5",
    cache_dir: str = "~/.cache/voxio-tts",
    **kwargs,
) -> CachedElevenLabsTTSService:
    """
    Create a cached ElevenLabs TTS service.
    
    Drop-in replacement for ElevenLabsTTSService with transparent caching.
    """
    return CachedElevenLabsTTSService(
        api_key=api_key,
        voice_id=voice_id,
        model=model,
        cache_dir=cache_dir,
        **kwargs,
    )
