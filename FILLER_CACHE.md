# Filler Response Pre-Caching System

Hide LLM latency by playing pre-cached filler responses ("Hmm", "Got it", etc.) while the real response is being generated.

## Features

- ✅ **Auto-caching**: Automatically detects and generates missing cache files
- ✅ **Non-blocking**: Bot starts immediately, caching happens in background
- ✅ **Hot-reload**: New fillers become available instantly as they're generated
- ✅ **Voice change support**: Auto-caches when switching to a new voice
- ✅ **Priority ordering**: Acknowledgments are cached first (most used)

## Overview

```
User: "What's the weather?"
          │
          ▼
    User stops speaking
          │
          ▼ (100ms)
    ┌─────────────────┐
    │ Play: "Mm-hmm." │ ◀── Cached, instant playback
    └─────────────────┘
          │
          ▼ (LLM thinking...)
          │
          ▼ (TTS generating...)
          │
          ▼
    ┌─────────────────────────────────────┐
    │ Play: "It's 72 degrees and sunny." │ ◀── Real response
    └─────────────────────────────────────┘
```

## Quick Start

### Option A: Auto-Caching (Recommended)

Just add FillerInjector with your API key - it handles everything automatically:

```python
from filler_cache import FillerInjector

filler_injector = FillerInjector(
    voice_id=TTS_VOICE,
    api_key=os.getenv("ELEVENLABS_API_KEY"),  # Enables auto-caching
    auto_cache=True,                           # Default: True
)

# Add to pipeline (before LLM)
pipeline = Pipeline([stt, filler_injector, llm, tts, output])
```

**What happens:**
1. Bot starts immediately (no waiting)
2. Loads any existing cached fillers
3. Detects missing fillers and generates them in background
4. Hot-loads new fillers as they become available
5. On voice change, auto-caches for new voice

### Option B: Manual Pre-Caching

Pre-generate cache before starting the bot:

```bash
# Use voice from config.toml
python precache_fillers.py

# Or specify voice ID
python precache_fillers.py --voice-id QzTKubutNn9TjrB7Xb2Q

# Pre-cache for multiple voices
python precache_fillers.py -v voice1 -v voice2 -v voice3
```

Then use FillerInjector without API key:

```python
filler_injector = FillerInjector(
    voice_id=TTS_VOICE,
    play_acknowledgment=True,
    play_thinking=False,
    acknowledgment_delay_ms=100,
)
```

## Filler Phrases

### Acknowledgments (played after user speaks)
- "Hmm."
- "Mm-hmm."
- "Right."
- "I see."
- "Okay."
- "Got it."
- "Ah."
- "Interesting."
- "Sure."
- "Yeah."

### Thinking (played while waiting - optional)
- "Let me think..."
- "One moment..."
- "Let me check..."
- "Hmm, let's see..."
- "Give me a sec..."

### Transitions (before main response)
- "So,"
- "Well,"
- "Alright,"
- "Okay, so"

## Configuration

### FillerInjector Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voice_id` | required | Voice ID for cached fillers |
| `play_acknowledgment` | `True` | Play "Hmm", "Got it" after user |
| `play_thinking` | `False` | Play "Let me think..." while waiting |
| `acknowledgment_delay_ms` | `100` | Delay before acknowledgment |
| `thinking_delay_ms` | `1500` | Delay before thinking filler |

### config.toml (optional)

```toml
[fillers]
enabled = true
play_acknowledgment = true
play_thinking = false
acknowledgment_delay_ms = 100
thinking_delay_ms = 1500
cache_dir = "~/.cache/voxio-fillers"
```

## How It Works

### Pre-caching

1. Each phrase is generated via ElevenLabs API
2. Audio is saved as raw PCM at `~/.cache/voxio-fillers/{voice_id}/{hash}.pcm`
3. On bot startup, all cached audio is loaded into memory

### Runtime

1. **User stops speaking** → FillerInjector starts timer
2. **100ms later** → Play random acknowledgment filler
3. **LLM starts responding** → Cancel any pending fillers
4. **TTS streams** → Real response plays normally

### Cancellation

Fillers are automatically cancelled when:
- User starts speaking again
- LLM response begins
- Real TTS audio starts

## File Structure

```
~/.cache/voxio-fillers/
├── QzTKubutNn9TjrB7Xb2Q/          # Voice ID
│   ├── a1b2c3d4e5f6g7h8.pcm       # "Hmm."
│   ├── b2c3d4e5f6g7h8i9.pcm       # "Got it."
│   └── ...
├── CwhRBWXzGAHq8TQ4Fs17/          # Another voice
│   └── ...
```

## API Usage

### Pre-cache programmatically

```python
from filler_cache import precache_fillers, FillerConfig

config = FillerConfig(cache_dir="~/.cache/voxio-fillers")

await precache_fillers(
    api_key="your_api_key",
    voice_id="QzTKubutNn9TjrB7Xb2Q",
    config=config,
)
```

### Load and play fillers manually

```python
from filler_cache import load_filler_audio, FillerConfig

config = FillerConfig()
audio = await load_filler_audio("Got it.", voice_id, config)
# audio is raw PCM bytes
```

## Benefits

| Metric | Without Fillers | With Fillers |
|--------|-----------------|--------------|
| Perceived latency | 1-3 seconds | ~100ms |
| User experience | Awkward silence | Natural conversation |
| API calls | Same | Same (fillers are cached) |

## Customization

### Add custom phrases

Edit `filler_cache.py`:

```python
FILLERS_ACKNOWLEDGMENT = [
    "Hmm.",
    "Got it.",
    "Your custom phrase.",  # Add here
]
```

Then re-run pre-caching:
```bash
python precache_fillers.py --voice-id your_voice
```

### Voice-specific phrases

Create different filler sets for different personas:

```python
FILLERS_FORMAL = ["I understand.", "Certainly.", "Of course."]
FILLERS_CASUAL = ["Yeah.", "Sure thing.", "Cool."]
```

## Troubleshooting

### Fillers not playing

1. Check cache exists: `ls ~/.cache/voxio-fillers/{voice_id}/`
2. Ensure FillerInjector is in pipeline before LLM
3. Check logs for "Preloaded X fillers into memory"

### Fillers cut off by response

Increase `acknowledgment_delay_ms` to give filler more time to finish.

### Want different timing

Adjust delays:
```python
FillerInjector(
    voice_id=voice_id,
    acknowledgment_delay_ms=200,  # Longer pause before filler
    thinking_delay_ms=2000,       # Longer wait before "thinking" filler
)
```
