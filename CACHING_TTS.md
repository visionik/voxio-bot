# ElevenLabs TTS Caching

This document describes the TTS caching system for Voxio Bot.

## Overview

The `CachedElevenLabsTTSService` wraps the standard ElevenLabs TTS service with transparent caching:

- **First request**: Streams audio immediately from ElevenLabs while saving chunks to cache
- **Subsequent requests**: Streams directly from local cache (no API call)

This saves API costs and reduces latency for repeated phrases.

## How It Works

```
Text Input → MD5(text + voice_id + model) → Cache Key
                                              ↓
                                    ┌─────────┴─────────┐
                                    │   Cache exists?   │
                                    └─────────┬─────────┘
                              ┌───────────────┼───────────────┐
                              ↓ NO            ↓               ↓ YES
                    ┌─────────────────┐                ┌──────────────┐
                    │ Call ElevenLabs │                │ Read from    │
                    │ Stream + Cache  │                │ cache file   │
                    └─────────────────┘                └──────────────┘
                              ↓                               ↓
                        Stream audio frames to pipeline
```

## Integration

### Option 1: Modify create_tts_service() in bot.py

Replace the ElevenLabs section in `create_tts_service()`:

```python
from cached_tts import CachedElevenLabsTTSService

def create_tts_service():
    provider = TTS_PROVIDER
    voice = TTS_VOICE
    
    if provider == "elevenlabs":
        voice_ids = {
            "roger": "CwhRBWXzGAHq8TQ4Fs17",
            # ... other voices
        }
        voice_id = voice_ids.get(voice.lower(), voice)
        
        logger.info(f"🔊 Using ElevenLabs TTS (cached) - voice: {voice}")
        return CachedElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=voice_id,
            model="eleven_turbo_v2_5",
            cache_dir="~/.cache/voxio-tts",
            cache_max_age_days=30,
        )
    # ... rest of function
```

### Option 2: Config-driven caching

Add to `config.toml`:

```toml
[tts]
provider = "elevenlabs"
voice = "QzTKubutNn9TjrB7Xb2Q"
cache_enabled = true
cache_dir = "~/.cache/voxio-tts"
cache_max_age_days = 30
```

Then in bot.py:

```python
def get_tts_cache_enabled():
    return get_config("tts.cache_enabled", False)

def get_tts_cache_dir():
    return get_config("tts.cache_dir", "~/.cache/voxio-tts")

def create_tts_service():
    if provider == "elevenlabs":
        if get_tts_cache_enabled():
            from cached_tts import CachedElevenLabsTTSService
            return CachedElevenLabsTTSService(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=voice_id,
                model="eleven_turbo_v2_5",
                cache_dir=get_tts_cache_dir(),
            )
        else:
            return ElevenLabsTTSService(...)
```

## Cache Location

Default: `~/.cache/voxio-tts/`

Files are named `{md5_hash}.pcm` and contain raw PCM audio data.

## Cache Management

### View stats
```python
tts.get_cache_stats()
# {'hits': 42, 'misses': 10, 'hit_rate': 0.807, 'cache_dir': '/Users/...'}
```

### Clear cache
```python
await tts.clear_cache()
```

### Automatic cleanup
Old files are cleaned up based on `cache_max_age_days` setting.

## Good Candidates for Caching

These phrases are repeated often and benefit from caching:

- Greetings ("Vinston here. What needs fixing?")
- Acknowledgments ("Got it.", "On it.", "Done.")
- Error messages ("I couldn't complete that request.")
- Common responses ("Let me check...", "One moment...")

## Limitations

1. **Voice changes**: Cache is keyed by voice_id, so changing voices invalidates cache
2. **Dynamic content**: Phrases with timestamps, names, etc. won't hit cache
3. **Disk space**: Long audio files use more space (estimate ~50KB per 10 seconds)

## Dependencies

Add `aiofiles` to requirements.txt:

```
aiofiles>=23.0.0
```

## Testing

```python
# Test cache behavior
tts = create_cached_tts_service(
    api_key="...",
    voice_id="...",
)

# First call - cache miss
async for frame in tts.run_tts("Hello world"):
    print(frame)

# Second call - cache hit (no API call)
async for frame in tts.run_tts("Hello world"):
    print(frame)

print(tts.get_cache_stats())
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, ...}
```
