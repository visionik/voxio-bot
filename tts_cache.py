"""
Unified TTS Cache for Voxio Bot.

Provides:
1. TTSCache - Core cache with get/put/precache
2. CachedTTSService - Drop-in ElevenLabs wrapper with caching
3. FillerInjector - Pipeline processor for filler playback

Usage:
    # Basic caching with TTS service
    tts = CachedTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="...",
    )
    
    # Pre-cache fillers at startup
    await tts.cache.precache(FILLER_PHRASES, voice_id)
    
    # Use FillerInjector in pipeline
    filler = FillerInjector(cache=tts.cache, voice_id="...")
    pipeline = Pipeline([stt, filler, llm, tts, output])
"""

import os
import asyncio
import hashlib
import random
import time
from pathlib import Path
from typing import Optional, AsyncGenerator, List, Dict, Any, Callable
from dataclasses import dataclass
from loguru import logger

import aiofiles
import aiofiles.os

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    AudioRawFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


# =============================================================================
# FILLER PHRASES
# =============================================================================

FILLERS_ACKNOWLEDGMENT = [
    "Hmm.",
    "Right.",
    "I see.",
    "Okay.",
    "Got it.",
    "Interesting.",
    "Sure.",
]

FILLERS_THINKING = [
    "Let me think...",
    "One moment...",
    "Let me check...",
    "Hmm, let's see...",
]

FILLERS_TRANSITION = [
    "So,",
    "Well,",
    "Alright,",
]

FILLER_CATEGORIES = {
    "ack": FILLERS_ACKNOWLEDGMENT,
    "think": FILLERS_THINKING,
    "transition": FILLERS_TRANSITION,
}

ALL_FILLERS = FILLERS_ACKNOWLEDGMENT + FILLERS_THINKING + FILLERS_TRANSITION


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

def load_config_toml(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config.toml and return cache section."""
    import tomllib
    
    if config_path is None:
        config_path = Path(__file__).parent / "config.toml"
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        return config.get("cache", {})
    except Exception as e:
        logger.warning(f"Failed to load config.toml: {e}")
        return {}


@dataclass
class CacheConfig:
    """
    Configuration for TTS cache.
    
    Loads defaults from config.toml [cache] section if present:
        [cache]
        dir = "~/.cache/voxio-tts"
        max_age_days = 30
        precache_fillers = true
        model = "eleven_turbo_v2_5"
    """
    cache_dir: str = "~/.cache/voxio-tts"
    model: str = "eleven_turbo_v2_5"
    sample_rate: int = 24000
    chunk_size: int = 4800  # 100ms at 24kHz mono 16-bit
    max_age_days: int = 30  # 0 = no cleanup
    precache_fillers: bool = True  # Auto-precache fillers on startup
    
    def __post_init__(self):
        self._expanded_dir: Optional[Path] = None
    
    @property
    def dir(self) -> Path:
        """Get expanded cache directory."""
        if self._expanded_dir is None:
            self._expanded_dir = Path(self.cache_dir).expanduser()
        return self._expanded_dir
    
    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "CacheConfig":
        """
        Load config from config.toml [cache] section.
        
        Falls back to defaults for missing values.
        """
        file_config = load_config_toml(config_path)
        
        return cls(
            cache_dir=file_config.get("dir", cls.cache_dir),
            model=file_config.get("model", cls.model),
            sample_rate=file_config.get("sample_rate", cls.sample_rate),
            chunk_size=file_config.get("chunk_size", cls.chunk_size),
            max_age_days=file_config.get("max_age_days", cls.max_age_days),
            precache_fillers=file_config.get("precache_fillers", cls.precache_fillers),
        )


# =============================================================================
# UNIFIED TTS CACHE
# =============================================================================

class TTSCache:
    """
    Unified cache for TTS audio.
    
    Features:
    - Async-safe with proper locking
    - Pre-cache support for fillers
    - Stream-while-save for cache misses
    - Automatic cleanup of old entries
    - Hit/miss statistics
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        api_key: Optional[str] = None,
    ):
        # Load from config.toml if no config provided
        self._config = config or CacheConfig.from_file()
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        
        # Ensure cache directory exists
        self._config.dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe state
        self._lock = asyncio.Lock()
        self._precache_tasks: Dict[str, asyncio.Task] = {}
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        # Callbacks for hot-reload
        self._on_cached_callbacks: List[Callable] = []
        
        logger.info(f"🗄️ TTS cache initialized: {self._config.dir}")
    
    # -------------------------------------------------------------------------
    # Cache Key Management
    # -------------------------------------------------------------------------
    
    def _get_key(self, text: str, voice_id: str) -> str:
        """Generate cache key from text + voice + model."""
        cache_input = f"{text}:{voice_id}:{self._config.model}"
        return hashlib.sha256(cache_input.encode()).hexdigest()[:16]
    
    def _get_path(self, key: str, voice_id: str) -> Path:
        """Get path to cache file."""
        voice_dir = self._config.dir / voice_id[:8]
        return voice_dir / f"{key}.pcm"
    
    # -------------------------------------------------------------------------
    # Core Cache Operations
    # -------------------------------------------------------------------------
    
    async def exists(self, text: str, voice_id: str) -> bool:
        """Check if audio is cached."""
        key = self._get_key(text, voice_id)
        path = self._get_path(key, voice_id)
        try:
            return await aiofiles.os.path.exists(path)
        except Exception:
            return False
    
    async def get(self, text: str, voice_id: str) -> Optional[bytes]:
        """
        Get cached audio.
        
        Returns:
            Audio bytes if cached, None otherwise.
        """
        key = self._get_key(text, voice_id)
        path = self._get_path(key, voice_id)
        
        try:
            if not await aiofiles.os.path.exists(path):
                return None
            
            async with aiofiles.open(path, 'rb') as f:
                data = await f.read()
            
            # Validate: must have some content
            if len(data) < 100:
                logger.warning(f"Cache file too small, removing: {path}")
                await self._safe_remove(path)
                return None
            
            async with self._lock:
                self._hits += 1
            
            logger.debug(f"🎯 Cache hit: {text[:20]}... ({len(data)} bytes)")
            return data
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    async def get_stream(
        self,
        text: str,
        voice_id: str,
    ) -> Optional[AsyncGenerator[bytes, None]]:
        """
        Get cached audio as a stream of chunks.
        
        Returns:
            Async generator of audio chunks, or None if not cached.
        """
        data = await self.get(text, voice_id)
        if data is None:
            return None
        
        async def _stream():
            chunk_size = self._config.chunk_size
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        
        return _stream()
    
    async def put(self, text: str, voice_id: str, audio: bytes) -> bool:
        """
        Save audio to cache.
        
        Returns:
            True if saved successfully.
        """
        if not audio or len(audio) < 100:
            logger.warning(f"Refusing to cache empty/tiny audio for: {text[:20]}...")
            return False
        
        key = self._get_key(text, voice_id)
        path = self._get_path(key, voice_id)
        
        try:
            # Ensure voice directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write via temp file
            temp_path = path.with_suffix('.tmp')
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(audio)
            
            await aiofiles.os.rename(temp_path, path)
            
            logger.debug(f"💾 Cached: {text[:20]}... ({len(audio)} bytes)")
            
            # Notify callbacks
            await self._notify_cached(text, voice_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            await self._safe_remove(path.with_suffix('.tmp'))
            return False
    
    async def _safe_remove(self, path: Path):
        """Safely remove a file."""
        try:
            await aiofiles.os.remove(path)
        except Exception:
            pass
    
    # -------------------------------------------------------------------------
    # Pre-caching
    # -------------------------------------------------------------------------
    
    async def precache(
        self,
        phrases: List[str],
        voice_id: str,
        priority_first: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Pre-generate and cache phrases.
        
        Args:
            phrases: List of phrases to cache
            voice_id: Voice ID to generate with
            priority_first: Phrases to generate first (e.g., acknowledgments)
        
        Returns:
            Dict mapping phrase to success status
        """
        if not self._api_key:
            logger.error("No API key for pre-caching")
            return {p: False for p in phrases}
        
        import aiohttp
        
        # Reorder for priority
        if priority_first:
            priority_set = set(priority_first)
            phrases = (
                [p for p in phrases if p in priority_set] +
                [p for p in phrases if p not in priority_set]
            )
        
        results = {}
        cached_count = 0
        generated_count = 0
        
        logger.info(f"🎙️ Pre-caching {len(phrases)} phrases for voice {voice_id[:8]}...")
        
        # ElevenLabs API endpoint (output_format is a query param, not body param)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_24000"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        
        async with aiohttp.ClientSession() as session:
            for phrase in phrases:
                # Skip if already cached
                if await self.exists(phrase, voice_id):
                    results[phrase] = True
                    cached_count += 1
                    continue
                
                try:
                    # Generate audio via ElevenLabs API
                    payload = {
                        "text": phrase,
                        "model_id": self._config.model,
                    }
                    
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f"ElevenLabs API error ({resp.status}): {error_text[:100]}")
                            results[phrase] = False
                            continue
                        
                        audio_data = await resp.read()
                    
                    # Save to cache
                    success = await self.put(phrase, voice_id, audio_data)
                    results[phrase] = success
                    
                    if success:
                        generated_count += 1
                    
                    # Rate limit protection
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Pre-cache failed for '{phrase}': {e}")
                    results[phrase] = False
        
        logger.info(
            f"✅ Pre-cache complete: {generated_count} generated, "
            f"{cached_count} already cached, "
            f"{len(phrases) - generated_count - cached_count} failed"
        )
        
        return results
    
    async def precache_background(
        self,
        phrases: List[str],
        voice_id: str,
        priority_first: Optional[List[str]] = None,
    ) -> bool:
        """
        Start pre-caching in background (non-blocking).
        
        Returns:
            True if started, False if already running for this voice.
        """
        if voice_id in self._precache_tasks:
            task = self._precache_tasks[voice_id]
            if not task.done():
                logger.debug(f"Pre-cache already running for {voice_id[:8]}...")
                return False
        
        async def _run():
            try:
                await self.precache(phrases, voice_id, priority_first)
            finally:
                self._precache_tasks.pop(voice_id, None)
        
        self._precache_tasks[voice_id] = asyncio.create_task(_run())
        return True
    
    async def precache_fillers(self, voice_id: str, background: bool = True) -> Any:
        """
        Convenience method to pre-cache all filler phrases.
        
        Args:
            voice_id: Voice ID
            background: Run in background (non-blocking)
        
        Returns:
            If background=True: bool (started)
            If background=False: Dict[str, bool] (results)
        """
        if background:
            return await self.precache_background(
                ALL_FILLERS,
                voice_id,
                priority_first=FILLERS_ACKNOWLEDGMENT,
            )
        else:
            return await self.precache(
                ALL_FILLERS,
                voice_id,
                priority_first=FILLERS_ACKNOWLEDGMENT,
            )
    
    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def on_cached(self, callback: Callable):
        """Register callback for when new items are cached."""
        self._on_cached_callbacks.append(callback)
    
    async def _notify_cached(self, text: str, voice_id: str):
        """Notify callbacks that an item was cached."""
        for cb in self._on_cached_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(text, voice_id)
                else:
                    cb(text, voice_id)
            except Exception as e:
                logger.error(f"Cache callback error: {e}")
    
    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------
    
    async def cleanup(self, max_age_days: Optional[int] = None) -> int:
        """
        Remove cache files older than max_age_days.
        
        Returns:
            Number of files removed.
        """
        max_age = max_age_days or self._config.max_age_days
        if max_age <= 0:
            return 0
        
        max_age_secs = max_age * 24 * 60 * 60
        now = time.time()
        removed = 0
        
        try:
            for voice_dir in self._config.dir.iterdir():
                if not voice_dir.is_dir():
                    continue
                for cache_file in voice_dir.glob("*.pcm"):
                    try:
                        stat = await aiofiles.os.stat(cache_file)
                        if now - stat.st_mtime > max_age_secs:
                            await aiofiles.os.remove(cache_file)
                            removed += 1
                    except Exception:
                        pass
            
            if removed:
                logger.info(f"🧹 Cleaned up {removed} old cache files")
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
        
        return removed
    
    async def clear(self, voice_id: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            voice_id: If provided, clear only this voice's cache
        
        Returns:
            Number of files removed.
        """
        removed = 0
        
        try:
            if voice_id:
                voice_dir = self._config.dir / voice_id[:8]
                if voice_dir.exists():
                    for f in voice_dir.glob("*.pcm"):
                        await aiofiles.os.remove(f)
                        removed += 1
            else:
                for voice_dir in self._config.dir.iterdir():
                    if voice_dir.is_dir():
                        for f in voice_dir.glob("*.pcm"):
                            await aiofiles.os.remove(f)
                            removed += 1
            
            logger.info(f"🗑️ Cleared {removed} cache files")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
        
        return removed
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, total),
            "total_requests": total,
            "cache_dir": str(self._config.dir),
        }


# =============================================================================
# CACHED TTS SERVICE
# =============================================================================

class CachedTTSService(ElevenLabsTTSService):
    """
    ElevenLabs TTS service with transparent caching.
    
    On cache hit: Streams from cache.
    On cache miss: Streams from ElevenLabs while saving to cache.
    """
    
    def __init__(
        self,
        *,
        cache: Optional[TTSCache] = None,
        cache_config: Optional[CacheConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self._voice_id = kwargs.get("voice_id", "")
        api_key = kwargs.get("api_key")
        
        # Use provided cache or create one (loads from config.toml)
        if cache:
            self.cache = cache
        else:
            if cache_config:
                config = cache_config
            else:
                config = CacheConfig.from_file()
                # Override model if specified in kwargs
                if "model" in kwargs:
                    config.model = kwargs["model"]
            self.cache = TTSCache(config=config, api_key=api_key)
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech, using cache when available."""
        
        # Try cache first
        stream = await self.cache.get_stream(text, self._voice_id)
        
        if stream is not None:
            # Cache hit - stream from cache
            yield TTSStartedFrame()
            
            async for chunk in stream:
                yield OutputAudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
            
            yield TTSStoppedFrame()
            return
        
        # Cache miss - generate and cache
        async with self.cache._lock:
            self.cache._misses += 1
        
        logger.debug(f"📝 Cache miss: {text[:30]}...")
        
        # Collect audio while streaming
        audio_chunks: List[bytes] = []
        
        async for frame in super().run_tts(text):
            if isinstance(frame, AudioRawFrame):
                audio_chunks.append(frame.audio)
            yield frame
        
        # Save to cache in background
        if audio_chunks:
            audio_data = b''.join(audio_chunks)
            asyncio.create_task(self.cache.put(text, self._voice_id, audio_data))
    
    async def set_voice(self, voice_id: str):
        """Change voice and optionally pre-cache fillers."""
        self._voice_id = voice_id
        # Could trigger precache here if desired


# =============================================================================
# FILLER INJECTOR
# =============================================================================

class FillerManager:
    """
    Event-based filler manager that injects audio directly to transport.
    
    Unlike FillerInjector (FrameProcessor), this doesn't sit in the pipeline.
    Instead, it's triggered by task event handlers and writes audio directly
    to the output transport.
    """
    
    def __init__(
        self,
        transport_output,  # SmallWebRTCOutputTransport
        cache: TTSCache,
        voice_id: str,
        *,
        play_acknowledgment: bool = True,
        play_thinking: bool = True,
        ack_delay: float = 0.4,
        think_delay: float = 2.5,
        sample_rate: int = 24000,
    ):
        self._transport = transport_output
        self._cache = cache
        self._voice_id = voice_id
        self._play_ack = play_acknowledgment
        self._play_think = play_thinking
        self._ack_delay = ack_delay
        self._think_delay = think_delay
        self._sample_rate = sample_rate
        
        self._loaded: dict[str, bytes] = {}
        self._filler_task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._state_lock = asyncio.Lock()
        self._initialized = False
        self._last_filler: dict[str, str] = {}  # Track last filler per category to avoid repeats
        
        logger.info(f"🎭 FillerManager initialized (voice: {voice_id[:8]}...)")
    
    async def initialize(self):
        """Load fillers - call this after construction."""
        if self._initialized:
            return
        self._initialized = True
        await self._load_fillers()
        self._cache.on_cached(self._on_new_cache)
    
    async def _load_fillers(self):
        """Load all cached fillers into memory."""
        loaded = 0
        for phrase in ALL_FILLERS:
            audio = await self._cache.get(phrase, self._voice_id)
            if audio:
                self._loaded[phrase] = audio
                loaded += 1
        logger.info(f"📦 FillerManager loaded {loaded}/{len(ALL_FILLERS)} fillers")
    
    async def _on_new_cache(self, text: str, voice_id: str):
        """Hot-reload newly cached fillers."""
        if voice_id != self._voice_id or text not in ALL_FILLERS or text in self._loaded:
            return
        audio = await self._cache.get(text, self._voice_id)
        if audio:
            self._loaded[text] = audio
            logger.debug(f"🔥 Hot-loaded filler: {text}")
    
    def _get_random_filler(self, category: str) -> Optional[tuple[str, bytes]]:
        """Get a random loaded filler from category, avoiding last played."""
        phrases = FILLER_CATEGORIES.get(category, [])
        available = [(p, self._loaded[p]) for p in phrases if p in self._loaded]
        if not available:
            return None
        # Avoid repeating last filler in this category
        last = self._last_filler.get(category)
        if last and len(available) > 1:
            available = [(p, a) for p, a in available if p != last]
        chosen = random.choice(available)
        self._last_filler[category] = chosen[0]
        return chosen
    
    async def _send_audio(self, audio: bytes):
        """Send audio directly to transport."""
        chunk_size = self._cache._config.chunk_size
        
        for i in range(0, len(audio), chunk_size):
            async with self._state_lock:
                if self._cancelled:
                    return
            
            frame = OutputAudioRawFrame(
                audio=audio[i:i + chunk_size],
                sample_rate=self._sample_rate,
                num_channels=1,
            )
            await self._transport.send_audio(frame)
            await asyncio.sleep(0)  # Yield to allow cancellation
    
    async def _filler_sequence(self):
        """Play filler sequence while waiting for LLM."""
        try:
            # 1. Acknowledgment after short delay
            if self._play_ack:
                await asyncio.sleep(self._ack_delay)
                
                async with self._state_lock:
                    if self._cancelled:
                        return
                
                filler = self._get_random_filler("ack")
                if filler:
                    phrase, audio = filler
                    await self._send_audio(audio)
                    logger.debug(f"🎭 Played ack: '{phrase}'")
            
            # 2. Thinking after longer delay
            if self._play_think:
                await asyncio.sleep(self._think_delay)
                
                async with self._state_lock:
                    if self._cancelled:
                        return
                
                filler = self._get_random_filler("think")
                if filler:
                    phrase, audio = filler
                    await self._send_audio(audio)
                    logger.debug(f"🎭 Played think: '{phrase}'")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Filler sequence error: {e}")
    
    async def on_user_stopped_speaking(self):
        """Called when user stops speaking - start filler sequence."""
        await self.cancel()
        async with self._state_lock:
            self._cancelled = False
        self._filler_task = asyncio.create_task(self._filler_sequence())
        logger.debug("🎭 Filler sequence started")
    
    async def cancel(self):
        """Cancel any running filler playback."""
        async with self._state_lock:
            self._cancelled = True
        
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            try:
                await self._filler_task
            except asyncio.CancelledError:
                pass
            self._filler_task = None


class FillerInjector(FrameProcessor):
    """
    Pipeline processor that plays cached filler audio while waiting for LLM.
    
    Uses the unified TTSCache for filler audio.
    """
    
    def __init__(
        self,
        cache: TTSCache,
        voice_id: str,
        play_acknowledgment: bool = True,
        play_thinking: bool = False,
        ack_delay_ms: int = 100,
        think_delay_ms: int = 1500,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self._cache = cache
        self._voice_id = voice_id
        self._play_ack = play_acknowledgment
        self._play_think = play_thinking
        self._ack_delay = ack_delay_ms / 1000.0
        self._think_delay = think_delay_ms / 1000.0
        self._sample_rate = sample_rate
        
        # State with lock for thread safety
        self._state_lock = asyncio.Lock()
        self._filler_task: Optional[asyncio.Task] = None
        self._cancelled = False
        
        # Pre-loaded fillers for instant access
        self._loaded: Dict[str, bytes] = {}
        
        # Track last filler per category to avoid repeats
        self._last_filler: Dict[str, str] = {}
        
        # Track if we've initialized
        self._initialized = False
        
        logger.info(f"🎭 FillerInjector initialized (voice: {voice_id[:8]}...)")
    
    async def _load_fillers(self):
        """Load all cached fillers into memory."""
        loaded = 0
        for phrase in ALL_FILLERS:
            audio = await self._cache.get(phrase, self._voice_id)
            if audio:
                self._loaded[phrase] = audio
                loaded += 1
        
        logger.info(f"📦 Loaded {loaded}/{len(ALL_FILLERS)} fillers into memory")
    
    async def _on_new_cache(self, text: str, voice_id: str):
        """Hot-reload newly cached fillers."""
        if voice_id != self._voice_id:
            return
        if text not in ALL_FILLERS:
            return
        if text in self._loaded:
            return
        
        audio = await self._cache.get(text, self._voice_id)
        if audio:
            self._loaded[text] = audio
            logger.debug(f"🔥 Hot-loaded filler: {text}")
    
    async def change_voice(self, voice_id: str):
        """Change to a different voice."""
        if voice_id == self._voice_id:
            return
        
        logger.info(f"🔄 Changing filler voice: {self._voice_id[:8]}... → {voice_id[:8]}...")
        self._voice_id = voice_id
        self._loaded.clear()
        await self._load_fillers()
    
    def _get_random_filler(self, category: str) -> Optional[tuple[str, bytes]]:
        """Get a random loaded filler from category, avoiding last played."""
        phrases = FILLER_CATEGORIES.get(category, [])
        available = [(p, self._loaded[p]) for p in phrases if p in self._loaded]
        if not available:
            return None
        # Avoid repeating last filler in this category
        last = self._last_filler.get(category)
        if last and len(available) > 1:
            available = [(p, a) for p, a in available if p != last]
        chosen = random.choice(available)
        self._last_filler[category] = chosen[0]
        return chosen
    
    async def _play_audio(self, audio: bytes):
        """Push audio frames to pipeline."""
        chunk_size = self._cache._config.chunk_size
        
        for i in range(0, len(audio), chunk_size):
            async with self._state_lock:
                if self._cancelled:
                    return
            
            yield OutputAudioRawFrame(
                audio=audio[i:i + chunk_size],
                sample_rate=self._sample_rate,
                num_channels=1,
            )
            await asyncio.sleep(0)  # Yield to allow cancellation
    
    async def _filler_sequence(self):
        """Play filler sequence while waiting for LLM."""
        try:
            # 1. Acknowledgment after short delay
            if self._play_ack:
                await asyncio.sleep(self._ack_delay)
                
                async with self._state_lock:
                    if self._cancelled:
                        return
                
                filler = self._get_random_filler("ack")
                if filler:
                    phrase, audio = filler
                    # Don't push TTS frames - just inject audio directly
                    async for frame in self._play_audio(audio):
                        await self.push_frame(frame)
                    logger.debug(f"🎭 Played: '{phrase}'")
            
            # 2. Thinking after longer delay
            if self._play_think:
                await asyncio.sleep(self._think_delay)
                
                async with self._state_lock:
                    if self._cancelled:
                        return
                
                filler = self._get_random_filler("think")
                if filler:
                    phrase, audio = filler
                    # Don't push TTS frames - just inject audio directly
                    async for frame in self._play_audio(audio):
                        await self.push_frame(frame)
                    logger.debug(f"🎭 Played: '{phrase}'")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Filler sequence error: {e}")
    
    async def _cancel_filler(self):
        """Cancel any running filler playback."""
        async with self._state_lock:
            self._cancelled = True
        
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            try:
                await self._filler_task
            except asyncio.CancelledError:
                pass
            self._filler_task = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and inject fillers."""
        
        # Debug: log all frames passing through
        frame_name = type(frame).__name__
        if "LLM" in frame_name or "Context" in frame_name or "Message" in frame_name:
            logger.debug(f"🎭 FillerInjector received: {frame_name}")
        
        # Initialize on StartFrame (load fillers into memory)
        if isinstance(frame, StartFrame) and not self._initialized:
            self._initialized = True
            await self._load_fillers()
            self._cache.on_cached(self._on_new_cache)
        
        # User stopped speaking → start filler
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Cancel any existing filler first to avoid overlapping playback
            await self._cancel_filler()
            async with self._state_lock:
                self._cancelled = False
            self._filler_task = asyncio.create_task(self._filler_sequence())
        
        # User started speaking or LLM responding → cancel filler
        elif isinstance(frame, (UserStartedSpeakingFrame, LLMFullResponseStartFrame)):
            await self._cancel_filler()
        
        # Let parent handle system frames (StartFrame, etc.) and pass through
        await super().process_frame(frame, direction)


# =============================================================================
# CLI
# =============================================================================

def load_voice_from_config() -> Optional[str]:
    """Load voice ID from config.toml [tts] section."""
    import tomllib
    config_path = Path(__file__).parent / "config.toml"
    if config_path.exists():
        try:
            with open(config_path, 'rb') as f:
                config = tomllib.load(f)
            return config.get("tts", {}).get("voice")
        except Exception:
            pass
    return None


async def cli_main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voxio TTS Cache Manager")
    parser.add_argument("action", choices=["precache", "list", "stats", "clear", "cleanup", "config"])
    parser.add_argument("--voice-id", "-v", help="Voice ID (default: from config.toml)")
    parser.add_argument("--api-key", "-k", default=os.getenv("ELEVENLABS_API_KEY"))
    parser.add_argument("--cache-dir", "-d", help="Cache directory (default: from config.toml)")
    parser.add_argument("--max-age", type=int, help="Max age for cleanup (default: from config.toml)")
    
    args = parser.parse_args()
    
    # Load config from file, override with CLI args
    config = CacheConfig.from_file()
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.max_age is not None:
        config.max_age_days = args.max_age
    
    # Get voice ID from args or config
    voice_id = args.voice_id or load_voice_from_config()
    
    cache = TTSCache(config=config, api_key=args.api_key)
    
    if args.action == "config":
        print("Cache configuration:")
        print(f"  dir:              {config.cache_dir}")
        print(f"  max_age_days:     {config.max_age_days}")
        print(f"  precache_fillers: {config.precache_fillers}")
        print(f"  model:            {config.model}")
        print(f"  sample_rate:      {config.sample_rate}")
        print(f"  chunk_size:       {config.chunk_size}")
        print(f"  voice (from tts): {voice_id or '(not set)'}")
    
    elif args.action == "list":
        print("Filler phrases:")
        for cat, phrases in FILLER_CATEGORIES.items():
            print(f"\n  {cat}:")
            for p in phrases:
                cached = "✓" if voice_id and await cache.exists(p, voice_id) else " "
                print(f"    [{cached}] {p}")
        if not voice_id:
            print("\n  (provide --voice-id to check cache status)")
    
    elif args.action == "precache":
        if not voice_id:
            print("Error: --voice-id required (or set tts.voice in config.toml)")
            return
        if not args.api_key:
            print("Error: --api-key or ELEVENLABS_API_KEY required")
            return
        await cache.precache_fillers(voice_id, background=False)
    
    elif args.action == "stats":
        print(cache.stats())
    
    elif args.action == "clear":
        removed = await cache.clear(voice_id)
        print(f"Removed {removed} files")
    
    elif args.action == "cleanup":
        removed = await cache.cleanup()
        print(f"Removed {removed} old files")


if __name__ == "__main__":
    asyncio.run(cli_main())
