"""
Pre-cached filler responses for latency hiding.

This module provides:
1. Pre-generation of common filler phrases ("Hmm", "Got it", etc.)
2. A FillerInjector that plays cached fillers while LLM generates responses
3. Auto-caching: automatically detects and generates missing cache files

Usage:
    # Auto-caching mode (recommended) - caches in background if needed
    filler_injector = FillerInjector(
        voice_id=voice_id,
        api_key=os.getenv("ELEVENLABS_API_KEY"),  # Enables auto-caching
        auto_cache=True,
    )
    
    # Manual pre-generation
    await precache_fillers(api_key, voice_id)
"""

import os
import random
import asyncio
import hashlib
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


# =============================================================================
# FILLER PHRASES
# =============================================================================

# Short acknowledgments (played immediately after user stops speaking)
FILLERS_ACKNOWLEDGMENT = [
    "Hmm.",
    "Mm-hmm.",
    "Right.",
    "I see.",
    "Okay.",
    "Got it.",
    "Ah.",
    "Interesting.",
    "Sure.",
    "Yeah.",
]

# Thinking indicators (played while waiting for LLM)
FILLERS_THINKING = [
    "Let me think...",
    "One moment...",
    "Let me check...",
    "Hmm, let's see...",
    "Give me a sec...",
]

# Transition phrases (played before main response)
FILLERS_TRANSITION = [
    "So,",
    "Well,",
    "Alright,",
    "Okay, so",
]

# All fillers combined with categories
FILLER_CATEGORIES = {
    "ack": FILLERS_ACKNOWLEDGMENT,
    "think": FILLERS_THINKING,
    "transition": FILLERS_TRANSITION,
}

ALL_FILLERS = FILLERS_ACKNOWLEDGMENT + FILLERS_THINKING + FILLERS_TRANSITION


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

@dataclass
class FillerConfig:
    """Configuration for filler cache."""
    cache_dir: str = "~/.cache/voxio-fillers"
    model: str = "eleven_turbo_v2_5"
    sample_rate: int = 24000


def get_filler_cache_dir(config: FillerConfig) -> Path:
    """Get expanded cache directory path."""
    return Path(config.cache_dir).expanduser()


def get_filler_cache_key(text: str, voice_id: str, model: str) -> str:
    """Generate cache key for a filler phrase."""
    cache_input = f"filler:{text}:{voice_id}:{model}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def get_filler_cache_path(text: str, voice_id: str, config: FillerConfig) -> Path:
    """Get path to cached filler audio file."""
    cache_dir = get_filler_cache_dir(config)
    cache_key = get_filler_cache_key(text, voice_id, config.model)
    return cache_dir / voice_id / f"{cache_key}.pcm"


async def is_filler_cached(text: str, voice_id: str, config: FillerConfig) -> bool:
    """Check if a filler phrase is already cached."""
    cache_path = get_filler_cache_path(text, voice_id, config)
    try:
        return await aiofiles.os.path.exists(cache_path)
    except Exception:
        return False


async def load_filler_audio(text: str, voice_id: str, config: FillerConfig) -> Optional[bytes]:
    """Load cached filler audio from disk."""
    cache_path = get_filler_cache_path(text, voice_id, config)
    try:
        async with aiofiles.open(cache_path, 'rb') as f:
            return await f.read()
    except Exception as e:
        logger.warning(f"Failed to load filler '{text}': {e}")
        return None


async def save_filler_audio(text: str, voice_id: str, audio: bytes, config: FillerConfig):
    """Save filler audio to cache."""
    cache_path = get_filler_cache_path(text, voice_id, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = cache_path.with_suffix('.tmp')
    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(audio)
        await aiofiles.os.rename(temp_path, cache_path)
        logger.debug(f"💾 Cached filler: '{text}' ({len(audio)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save filler '{text}': {e}")
        try:
            await aiofiles.os.remove(temp_path)
        except Exception:
            pass


# =============================================================================
# PRE-CACHE GENERATION
# =============================================================================

async def precache_fillers(
    api_key: str,
    voice_id: str,
    config: Optional[FillerConfig] = None,
    phrases: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Pre-generate and cache all filler phrases for a voice.
    
    Args:
        api_key: ElevenLabs API key
        voice_id: Voice ID to generate for
        config: Cache configuration
        phrases: List of phrases to cache (default: ALL_FILLERS)
    
    Returns:
        Dict mapping phrase to success status
    """
    from elevenlabs import ElevenLabs
    
    config = config or FillerConfig()
    phrases = phrases or ALL_FILLERS
    
    # Ensure cache directory exists
    cache_dir = get_filler_cache_dir(config)
    voice_dir = cache_dir / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    client = ElevenLabs(api_key=api_key)
    results = {}
    
    logger.info(f"🎙️ Pre-caching {len(phrases)} fillers for voice {voice_id[:8]}...")
    
    for phrase in phrases:
        # Skip if already cached
        if await is_filler_cached(phrase, voice_id, config):
            logger.debug(f"⏭️ Already cached: '{phrase}'")
            results[phrase] = True
            continue
        
        try:
            # Generate audio
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                text=phrase,
                model_id=config.model,
                output_format="pcm_24000",
            )
            
            # Collect all chunks
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk)
            audio_data = b''.join(audio_chunks)
            
            # Save to cache
            await save_filler_audio(phrase, voice_id, audio_data, config)
            results[phrase] = True
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to generate filler '{phrase}': {e}")
            results[phrase] = False
    
    cached = sum(1 for v in results.values() if v)
    logger.info(f"✅ Pre-cached {cached}/{len(phrases)} fillers for voice {voice_id[:8]}")
    
    return results


async def precache_fillers_multi_voice(
    api_key: str,
    voice_ids: List[str],
    config: Optional[FillerConfig] = None,
):
    """Pre-cache fillers for multiple voices."""
    config = config or FillerConfig()
    
    for voice_id in voice_ids:
        await precache_fillers(api_key, voice_id, config)


# =============================================================================
# AUTO-CACHING SYSTEM
# =============================================================================

async def get_missing_fillers(
    voice_id: str,
    config: Optional[FillerConfig] = None,
    phrases: Optional[List[str]] = None,
) -> List[str]:
    """
    Check which filler phrases are missing from cache for a voice.
    
    Returns:
        List of phrases that need to be generated
    """
    config = config or FillerConfig()
    phrases = phrases or ALL_FILLERS
    
    missing = []
    for phrase in phrases:
        if not await is_filler_cached(phrase, voice_id, config):
            missing.append(phrase)
    
    return missing


async def is_voice_fully_cached(
    voice_id: str,
    config: Optional[FillerConfig] = None,
) -> bool:
    """Check if all fillers are cached for a voice."""
    missing = await get_missing_fillers(voice_id, config)
    return len(missing) == 0


class AutoCacheManager:
    """
    Background manager for automatic filler caching.
    
    Monitors voice changes and automatically generates missing cache files
    in the background without blocking the main application.
    """
    
    _instance: Optional['AutoCacheManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(
        self,
        api_key: str,
        config: Optional[FillerConfig] = None,
    ):
        self._api_key = api_key
        self._config = config or FillerConfig()
        self._cache_tasks: Dict[str, asyncio.Task] = {}
        self._cached_voices: set = set()
        self._callbacks: List[callable] = []
    
    @classmethod
    async def get_instance(
        cls,
        api_key: Optional[str] = None,
        config: Optional[FillerConfig] = None,
    ) -> 'AutoCacheManager':
        """Get or create singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                if api_key is None:
                    api_key = os.getenv("ELEVENLABS_API_KEY")
                if not api_key:
                    raise ValueError("ELEVENLABS_API_KEY required for auto-caching")
                cls._instance = cls(api_key, config)
            return cls._instance
    
    def on_cache_complete(self, callback: callable):
        """Register callback for when caching completes for a voice."""
        self._callbacks.append(callback)
    
    async def _notify_complete(self, voice_id: str, phrases: List[str]):
        """Notify callbacks that caching is complete."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(voice_id, phrases)
                else:
                    callback(voice_id, phrases)
            except Exception as e:
                logger.error(f"Cache callback error: {e}")
    
    async def ensure_cached(
        self,
        voice_id: str,
        priority_phrases: Optional[List[str]] = None,
    ) -> bool:
        """
        Ensure fillers are cached for a voice, generating in background if needed.
        
        Args:
            voice_id: Voice ID to cache
            priority_phrases: Phrases to cache first (e.g., acknowledgments)
        
        Returns:
            True if already cached, False if caching started in background
        """
        # Check if already fully cached
        if voice_id in self._cached_voices:
            return True
        
        missing = await get_missing_fillers(voice_id, self._config)
        
        if not missing:
            self._cached_voices.add(voice_id)
            logger.info(f"✅ Voice {voice_id[:8]}... already fully cached")
            return True
        
        # Check if already caching this voice
        if voice_id in self._cache_tasks:
            task = self._cache_tasks[voice_id]
            if not task.done():
                logger.debug(f"⏳ Already caching voice {voice_id[:8]}...")
                return False
        
        # Start background caching
        logger.info(f"🔄 Auto-caching {len(missing)} missing fillers for voice {voice_id[:8]}...")
        
        # Reorder to prioritize certain phrases
        if priority_phrases:
            priority_set = set(priority_phrases)
            missing_priority = [p for p in missing if p in priority_set]
            missing_other = [p for p in missing if p not in priority_set]
            missing = missing_priority + missing_other
        
        # Start background task
        task = asyncio.create_task(self._background_cache(voice_id, missing))
        self._cache_tasks[voice_id] = task
        
        return False
    
    async def _background_cache(self, voice_id: str, phrases: List[str]):
        """Background task to cache fillers."""
        try:
            results = await precache_fillers(
                self._api_key,
                voice_id,
                self._config,
                phrases=phrases,
            )
            
            # Mark as cached if all succeeded
            if all(results.values()):
                self._cached_voices.add(voice_id)
            
            # Notify listeners
            cached_phrases = [p for p, success in results.items() if success]
            await self._notify_complete(voice_id, cached_phrases)
            
        except Exception as e:
            logger.error(f"Background caching failed for {voice_id[:8]}: {e}")
        finally:
            # Clean up task reference
            if voice_id in self._cache_tasks:
                del self._cache_tasks[voice_id]
    
    def get_caching_status(self) -> Dict[str, Any]:
        """Get status of caching operations."""
        return {
            "cached_voices": list(self._cached_voices),
            "active_tasks": list(self._cache_tasks.keys()),
        }


# Global auto-cache manager instance
_auto_cache_manager: Optional[AutoCacheManager] = None


async def ensure_voice_cached(
    voice_id: str,
    api_key: Optional[str] = None,
    config: Optional[FillerConfig] = None,
) -> bool:
    """
    Convenience function to ensure a voice has cached fillers.
    
    Starts background caching if needed. Non-blocking.
    
    Returns:
        True if already cached, False if caching started
    """
    global _auto_cache_manager
    
    if _auto_cache_manager is None:
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("No API key for auto-caching fillers")
            return False
        _auto_cache_manager = AutoCacheManager(api_key, config)
    
    # Prioritize acknowledgment phrases (most commonly used)
    return await _auto_cache_manager.ensure_cached(
        voice_id,
        priority_phrases=FILLERS_ACKNOWLEDGMENT,
    )


# =============================================================================
# FILLER INJECTOR (Pipeline Processor)
# =============================================================================

class FillerInjector(FrameProcessor):
    """
    Pipeline processor that injects cached filler audio while waiting for LLM.
    
    Flow:
    1. User stops speaking → immediately play acknowledgment filler
    2. While LLM generates → optionally play thinking filler
    3. Real response starts → stop fillers, play actual response
    
    Features:
    - Auto-caching: Automatically generates missing cache files in background
    - Hot-reload: New fillers become available as they're generated
    - Non-blocking: Bot starts immediately, doesn't wait for cache
    
    Usage:
        filler_injector = FillerInjector(
            voice_id="...",
            api_key=os.getenv("ELEVENLABS_API_KEY"),  # Enables auto-cache
            auto_cache=True,
        )
        pipeline = Pipeline([stt, filler_injector, llm, tts, output])
    """
    
    def __init__(
        self,
        voice_id: str,
        config: Optional[FillerConfig] = None,
        play_acknowledgment: bool = True,
        play_thinking: bool = False,  # Can be annoying, disabled by default
        acknowledgment_delay_ms: int = 100,
        thinking_delay_ms: int = 1500,
        api_key: Optional[str] = None,
        auto_cache: bool = True,
        **kwargs,
    ):
        """
        Initialize filler injector.
        
        Args:
            voice_id: Voice ID for cached fillers
            config: Cache configuration
            play_acknowledgment: Play "Hmm", "Got it" after user speaks
            play_thinking: Play "Let me think..." while waiting
            acknowledgment_delay_ms: Delay before acknowledgment
            thinking_delay_ms: Delay before thinking filler
            api_key: ElevenLabs API key (enables auto-caching)
            auto_cache: Automatically cache missing fillers in background
        """
        super().__init__(**kwargs)
        
        self._voice_id = voice_id
        self._config = config or FillerConfig()
        self._play_acknowledgment = play_acknowledgment
        self._play_thinking = play_thinking
        self._ack_delay = acknowledgment_delay_ms / 1000.0
        self._think_delay = thinking_delay_ms / 1000.0
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self._auto_cache = auto_cache and bool(self._api_key)
        
        # State
        self._waiting_for_response = False
        self._filler_task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._cache_ready = False
        
        # Preload fillers into memory for instant playback
        self._loaded_fillers: Dict[str, bytes] = {}
        
        logger.info(f"🎭 FillerInjector initialized (voice: {voice_id[:8]}..., auto_cache: {self._auto_cache})")
    
    async def start(self, frame):
        """Load fillers into memory on pipeline start."""
        await super().start(frame)
        await self._preload_fillers()
        
        # Start auto-caching if enabled
        if self._auto_cache:
            await self._start_auto_cache()
    
    async def _preload_fillers(self):
        """Load all cached fillers into memory."""
        for phrase in ALL_FILLERS:
            audio = await load_filler_audio(phrase, self._voice_id, self._config)
            if audio:
                self._loaded_fillers[phrase] = audio
        
        loaded_count = len(self._loaded_fillers)
        total_count = len(ALL_FILLERS)
        
        if loaded_count == total_count:
            self._cache_ready = True
            logger.info(f"📦 Preloaded all {loaded_count} fillers into memory")
        elif loaded_count > 0:
            logger.info(f"📦 Preloaded {loaded_count}/{total_count} fillers (more caching in background)")
        else:
            logger.info(f"📦 No cached fillers found (will cache in background)")
    
    async def _start_auto_cache(self):
        """Start automatic background caching if needed."""
        if not self._api_key:
            logger.warning("⚠️ No API key provided, auto-caching disabled")
            return
        
        # Check what's missing
        missing = await get_missing_fillers(self._voice_id, self._config)
        
        if not missing:
            self._cache_ready = True
            logger.info(f"✅ All fillers already cached for voice {self._voice_id[:8]}...")
            return
        
        logger.info(f"🔄 Auto-caching {len(missing)} missing fillers in background...")
        
        # Get or create auto-cache manager
        global _auto_cache_manager
        if _auto_cache_manager is None:
            _auto_cache_manager = AutoCacheManager(self._api_key, self._config)
        
        # Register callback to hot-reload fillers as they're generated
        _auto_cache_manager.on_cache_complete(self._on_fillers_cached)
        
        # Start background caching (non-blocking)
        await _auto_cache_manager.ensure_cached(
            self._voice_id,
            priority_phrases=FILLERS_ACKNOWLEDGMENT,  # Cache these first
        )
    
    async def _on_fillers_cached(self, voice_id: str, phrases: List[str]):
        """Callback when new fillers are cached - hot-reload them."""
        if voice_id != self._voice_id:
            return
        
        # Load newly cached fillers into memory
        loaded = 0
        for phrase in phrases:
            if phrase not in self._loaded_fillers:
                audio = await load_filler_audio(phrase, self._voice_id, self._config)
                if audio:
                    self._loaded_fillers[phrase] = audio
                    loaded += 1
        
        if loaded > 0:
            logger.info(f"🔥 Hot-loaded {loaded} new fillers (total: {len(self._loaded_fillers)})")
        
        # Check if we're fully cached now
        if len(self._loaded_fillers) >= len(ALL_FILLERS):
            self._cache_ready = True
            logger.info(f"✅ All fillers now cached and loaded for voice {self._voice_id[:8]}...")
    
    async def change_voice(self, new_voice_id: str):
        """
        Change to a different voice, auto-caching if needed.
        
        Call this when the user changes voice to ensure fillers are cached.
        """
        if new_voice_id == self._voice_id:
            return
        
        logger.info(f"🔄 Changing filler voice: {self._voice_id[:8]}... → {new_voice_id[:8]}...")
        
        self._voice_id = new_voice_id
        self._loaded_fillers.clear()
        self._cache_ready = False
        
        # Reload for new voice
        await self._preload_fillers()
        
        if self._auto_cache:
            await self._start_auto_cache()
    
    def _get_random_filler(self, category: str) -> Optional[tuple[str, bytes]]:
        """Get a random filler from category that's loaded in memory."""
        phrases = FILLER_CATEGORIES.get(category, [])
        available = [(p, self._loaded_fillers[p]) for p in phrases if p in self._loaded_fillers]
        
        if not available:
            return None
        
        return random.choice(available)
    
    async def _play_filler(self, phrase: str, audio: bytes):
        """Push filler audio frames to pipeline."""
        # Push audio in chunks to match typical frame size
        chunk_size = 4800  # 100ms at 24kHz mono 16-bit
        
        await self.push_frame(TTSStartedFrame())
        
        for i in range(0, len(audio), chunk_size):
            if self._cancelled:
                break
            
            chunk = audio[i:i + chunk_size]
            await self.push_frame(AudioRawFrame(
                audio=chunk,
                sample_rate=self._config.sample_rate,
                num_channels=1,
            ))
            
            # Small yield to allow cancellation
            await asyncio.sleep(0)
        
        await self.push_frame(TTSStoppedFrame())
        logger.debug(f"🎭 Played filler: '{phrase}'")
    
    async def _filler_sequence(self):
        """Play filler sequence while waiting for LLM response."""
        try:
            # 1. Short delay then acknowledgment
            if self._play_acknowledgment:
                await asyncio.sleep(self._ack_delay)
                if self._cancelled:
                    return
                
                filler = self._get_random_filler("ack")
                if filler:
                    await self._play_filler(filler[0], filler[1])
            
            # 2. Longer delay then thinking indicator
            if self._play_thinking and not self._cancelled:
                await asyncio.sleep(self._think_delay)
                if self._cancelled:
                    return
                
                filler = self._get_random_filler("think")
                if filler:
                    await self._play_filler(filler[0], filler[1])
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Filler sequence error: {e}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and inject fillers at appropriate times."""
        
        # User stopped speaking → start filler sequence
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._waiting_for_response = True
            self._cancelled = False
            self._filler_task = asyncio.create_task(self._filler_sequence())
        
        # User started speaking → cancel any playing filler
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._cancel_filler()
        
        # LLM response starting → cancel filler, let real response through
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._cancel_filler()
            self._waiting_for_response = False
        
        # LLM response ended
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._waiting_for_response = False
        
        # Pass frame through
        await self.push_frame(frame, direction)
    
    def _cancel_filler(self):
        """Cancel any running filler playback."""
        self._cancelled = True
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            self._filler_task = None


# =============================================================================
# CLI FOR PRE-CACHING
# =============================================================================

async def cli_precache():
    """CLI entry point for pre-caching fillers."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-cache filler phrases for voice bot")
    parser.add_argument("--voice-id", required=True, help="ElevenLabs voice ID")
    parser.add_argument("--api-key", default=os.getenv("ELEVENLABS_API_KEY"), help="ElevenLabs API key")
    parser.add_argument("--cache-dir", default="~/.cache/voxio-fillers", help="Cache directory")
    parser.add_argument("--list", action="store_true", help="List all filler phrases")
    
    args = parser.parse_args()
    
    if args.list:
        print("Acknowledgment fillers:")
        for p in FILLERS_ACKNOWLEDGMENT:
            print(f"  - {p}")
        print("\nThinking fillers:")
        for p in FILLERS_THINKING:
            print(f"  - {p}")
        print("\nTransition fillers:")
        for p in FILLERS_TRANSITION:
            print(f"  - {p}")
        return
    
    if not args.api_key:
        print("Error: --api-key or ELEVENLABS_API_KEY required")
        return
    
    config = FillerConfig(cache_dir=args.cache_dir)
    await precache_fillers(args.api_key, args.voice_id, config)


if __name__ == "__main__":
    asyncio.run(cli_precache())
