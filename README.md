# Voxio Bot

A real-time voice AI assistant powered by [Clawdbot](https://github.com/clawdbot/clawdbot):
- **Brain**: [Clawdbot](https://clawdbot.com) (tools, memory, integrations)
- **STT**: MLX Whisper / OpenAI Whisper
- **Voice LLM**: Anthropic Claude (quick responses + handoff)
- **TTS**: ElevenLabs (with caching for reduced latency and cost)
- **Transport**: WebRTC / Daily.co
- **Framework**: [Pipecat](https://pipecat.ai)

## Features

- 🎙️ **Real-time voice conversation** with Claude
- 🔧 **Tool access** via Clawdbot (calendar, email, web search, etc.)
- 💾 **TTS caching** reduces latency and API costs
- 🗣️ **Filler responses** hide LLM latency ("Hmm", "Got it")
- 🎵 **Ambient sounds** during handoff (typing, office ambiance)
- 👁️ **Vision support** (local Moondream or Claude Vision)
- 🎬 **Video output** (avatar, GIFs, generated images)

## Architecture

![Voxio Architecture](docs/architecture.png)

## Quick Start

### 1. Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set up environment
```bash
cd voxio-bot
cp .env.example .env
# Edit .env with your API keys
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Run the bot (with auth)
```bash
uv run python run_auth.py --port 8086
```

### 5. Connect
Open http://localhost:8086/client in your browser and click **Connect**.

---

## Configuration

Voxio Bot uses a `config.toml` file for all settings. Copy from the example and customize:

```bash
cp config.toml.example config.toml
```

### Configuration Reference

#### `[llm]` - Language Model

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `mode` | `local`, `gateway` | `local` | LLM routing mode |

**Modes:**
- `local`: Voxio runs its own Claude with limited tools + handoff to Clawdbot
- `gateway`: All LLM calls route through Clawdbot Gateway (full tool access)

```toml
[llm]
mode = "local"
```

#### `[session]` - Clawdbot Session

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `key` | string | `"agent:main:voice"` | Session key for Clawdbot |
| `label` | string | - | Alternative: resolve session by label |
| `require_existing` | bool | `false` | Fail if session doesn't exist |
| `reset_on_connect` | bool | `false` | Fresh transcript on each connect |

```toml
[session]
key = "agent:main:voice"
require_existing = false
reset_on_connect = false
```

#### `[identity]` - Bot Personality

| Key | Type | Description |
|-----|------|-------------|
| `name` | string | Bot's display name |
| `greetings` | array | Random greeting on connect |

```toml
[identity]
name = "Vinston Wolf"
greetings = [
    "Vinston here. What needs fixing?",
    "I'm Vinston Wolf. I solve problems.",
    "Let's get down to brass tacks. How can I help?",
]
```

#### `[stt]` - Speech-to-Text

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `provider` | `openai`, `mlx-whisper`, `whisper` | `mlx-whisper` | STT provider |
| `model` | see below | `large-turbo` | Whisper model |

**Models:** `tiny`, `base`, `small`, `medium`, `large`, `large-turbo`, `large-turbo-q4`, `distil-large`

```toml
[stt]
provider = "mlx-whisper"  # Local on Apple Silicon
model = "large-turbo"     # Fast and accurate
```

#### `[tts]` - Text-to-Speech

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `elevenlabs`, `piper` | `elevenlabs` | TTS provider |
| `voice` | string | - | Voice ID (ElevenLabs) or model path (Piper) |

```toml
[tts]
provider = "elevenlabs"
voice = "QzTKubutNn9TjrB7Xb2Q"  # Custom voice ID
```

#### `[cache]` - TTS Caching

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dir` | string | `~/.cache/voxio-tts` | Cache directory |
| `max_age_days` | int | `30` | Auto-cleanup age (0 = disable) |
| `precache_fillers` | bool | `true` | Pre-generate filler phrases |

```toml
[cache]
dir = "~/.cache/voxio-tts"
max_age_days = 30
precache_fillers = true
```

#### `[handoff]` - Handoff Audio

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | `none`, `prompt`, `ambient` | `ambient` | Audio during handoff |
| `prompt` | string | - | SFX prompt (for `prompt` type) |
| `files` | array | - | Sound files (for `ambient` type) |
| `gap` | float | `0.5` | Gap between ambient sounds (seconds) |

```toml
[handoff]
type = "ambient"
files = [
    "sounds/handoff_typing.wav",
    "sounds/office-chair.wav",
]
gap = 0.5
```

#### `[video]` - Video Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_enabled` | bool | `true` | Enable camera input |
| `output_enabled` | bool | `true` | Enable video output |
| `avatar` | string | `avatar.png` | Default avatar image |
| `capture_display_duration` | int | `60` | Camera frame display (seconds) |
| `gif_display_duration` | int | `120` | GIF display duration |
| `image_display_duration` | int | `300` | Generated image duration |

```toml
[video]
input_enabled = true
output_enabled = true
avatar = "avatar.png"
capture_display_duration = 60
gif_display_duration = 120
image_display_duration = 300
```

#### `[vision]` - Vision Analysis

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `mode` | `handoff`, `direct`, `local` | `local` | Vision processing mode |
| `local_model` | `moondream`, `llava`, `florence-2` | `moondream` | Local model |

**Modes:**
- `handoff`: Send to Clawdbot for analysis
- `direct`: Feed to Claude Vision directly
- `local`: Use local model (Moondream on Apple Silicon)

```toml
[vision]
mode = "local"
local_model = "moondream"
```

#### `[vad]` - Voice Activity Detection

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `stop_secs` | float | `0.3` | Silence duration to end turn |

```toml
[vad]
stop_secs = 0.3  # 300ms silence = turn complete
```

#### `[transport]` - Connection Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | `webrtc`, `daily` | `webrtc` | Transport type |
| `port` | int | `8086` | Server port |
| `host` | string | `127.0.0.1` | Bind address |
| `proxy` | string | - | Public hostname (NAT traversal) |
| `esp32` | bool | `false` | ESP32 compatibility mode |
| `room` | string | - | Daily.co room URL |
| `verbose` | bool | `false` | Verbose logging |

```toml
[transport]
type = "webrtc"
port = 8086
verbose = false
```

---

## TTS Caching System

Voxio Bot includes a transparent caching layer for ElevenLabs TTS that reduces latency and API costs.

### How It Works

```
Text Input → MD5(text + voice + model) → Cache Key
                                           ↓
                               ┌───────────┴───────────┐
                               │    Cache exists?      │
                               └───────────┬───────────┘
                           NO ↓                        ↓ YES
                   ┌─────────────────┐      ┌──────────────────┐
                   │ Call ElevenLabs │      │ Read from cache  │
                   │ Stream + Cache  │      │ (instant!)       │
                   └─────────────────┘      └──────────────────┘
                               ↓                        ↓
                         Stream audio to user
```

### Features

- **Stream-first**: Audio plays immediately, caching happens in background
- **Transparent**: No changes needed to TTS calls
- **Automatic cleanup**: Old cache files removed based on `max_age_days`
- **Voice-aware**: Cache is keyed by voice ID, so voice changes work correctly

### Cache Location

```
~/.cache/voxio-tts/
├── {voice_id}/
│   ├── a1b2c3d4.pcm    # Cached phrase 1
│   ├── e5f6g7h8.pcm    # Cached phrase 2
│   └── ...
```

### Benefits

| Metric | Without Cache | With Cache |
|--------|---------------|------------|
| Latency | 200-500ms | <10ms |
| API Cost | Every request | First request only |
| Reliability | Network dependent | Local fallback |

---

## Filler Response System

Hide LLM latency by playing pre-cached filler responses while the real response is being generated.

### How It Works

```
User: "What's the weather?"
        │
        ▼ (User stops speaking)
        │
        ▼ (100ms delay)
┌───────────────────┐
│ Play: "Mm-hmm." 🎯 │ ◀── Cached, instant playback
└───────────────────┘
        │
        ▼ (LLM thinking... TTS generating...)
        │
┌─────────────────────────────────────────┐
│ Play: "It's 72 degrees and sunny."      │ ◀── Real response
└─────────────────────────────────────────┘
```

### Filler Phrases

**Acknowledgments** (played immediately after user speaks):
- "Hmm." / "Mm-hmm." / "Right." / "I see."
- "Okay." / "Got it." / "Ah." / "Sure." / "Yeah."

**Thinking** (played while waiting, optional):
- "Let me think..." / "One moment..." / "Let me check..."

### Auto-Caching

When `precache_fillers = true` (default), Voxio Bot:
1. Checks for missing filler audio on startup
2. Generates missing phrases in the background
3. Hot-loads new fillers as they become available
4. Re-caches automatically when voice changes

### Manual Pre-Caching

```bash
# Pre-cache for current voice
python precache_fillers.py

# Pre-cache for specific voice
python precache_fillers.py --voice-id QzTKubutNn9TjrB7Xb2Q

# List all filler phrases
python precache_fillers.py --list
```

### Configuration

```toml
[cache]
precache_fillers = true  # Enable auto-caching of fillers
```

---

## Ambient Sounds (Handoff Mode)

When Voxio Bot hands off a task to Clawdbot, it can play ambient sounds to fill the silence while waiting for a response.

### Handoff Types

| Type | Description |
|------|-------------|
| `none` | Silent waiting |
| `prompt` | Generate SFX via ElevenLabs API |
| `ambient` | Loop through sound files |

### Ambient Mode (Recommended)

Plays random sounds from a list with configurable gaps:

```toml
[handoff]
type = "ambient"
files = [
    "sounds/handoff_typing.wav",      # Keyboard typing
    "sounds/office-chair.wav",        # Chair movement
    "sounds/paper-shuffle.wav",       # Paper sounds
    "sounds/clearing-throat.wav",     # Human presence
]
gap = 0.5  # 500ms between sounds
```

### Included Sound Files

| File | Description |
|------|-------------|
| `handoff_typing.wav` | Keyboard typing sounds |
| `110451__freeborn__paper01.wav` | Paper shuffling |
| `377260__johnnypanic__clearing-throat-2.wav` | Throat clearing |
| `414819__bokal__office-drawer.wav` | Drawer opening |
| `403137__blouhond__office-chair.wav` | Office chair |
| `484659__inspectorj__cd-player.wav` | CD player |
| `60-writing.wav` | Pen writing |

### Character Sounds

For personality, you can add character-specific sounds:

```toml
files = [
    "sounds/winston-wolf.wav",        # Character clips
    "sounds/pulp-fiction-clip.wav",   # Movie quotes
]
```

### Adding Custom Sounds

1. Add `.wav` files to the `sounds/` directory
2. Update `config.toml` with the new file paths
3. Restart the bot

**Requirements:**
- Format: WAV (PCM)
- Sample rate: 24000 Hz (recommended)
- Channels: Mono

---

## Clawdbot Integration

Voxio Bot has bidirectional integration with Clawdbot:

### Handoff Tool (Voice → Clawdbot)

When users ask for something requiring tools (calendar, email, web search, etc.), Claude will use the `handoff_to_clawdbot` tool to delegate the task:

```
User: "What's on my calendar tomorrow?"
Claude: "Let me check that for you."
→ Plays ambient sounds while waiting
→ Triggers: clawdbot wake --mode now --text "[Voice Task] Check calendar..."
→ Clawdbot processes and responds via /speak endpoint
```

### /speak Endpoint (Clawdbot → Voice)

Clawdbot can speak results back through active voice sessions:

```bash
# Check active sessions
curl http://localhost:8086/sessions

# Speak to the active session
curl -X POST http://localhost:8086/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "You have two meetings tomorrow..."}'
```

**POST /speak**
```json
{
  "text": "Message to speak",
  "session_id": "optional - defaults to most recent session"
}
```

**GET /sessions**
```json
{
  "active_count": 1,
  "session_ids": ["abc123"],
  "default_session": "abc123"
}
```

### Complete Flow

1. User speaks: "Search for best restaurants nearby"
2. Voice Claude uses `handoff_to_clawdbot` tool
3. Ambient sounds play (typing, office ambiance)
4. Clawdbot performs web search
5. Clawdbot POSTs results to `/speak`
6. Voxio Bot speaks the results to the user

---

## API Keys Required

### Anthropic (Claude LLM)
1. Go to https://console.anthropic.com/
2. Create an account or sign in
3. Navigate to API Keys
4. Create a new key
5. Add to `.env` as `ANTHROPIC_API_KEY`

### OpenAI (Whisper STT) - Optional
1. Go to https://platform.openai.com/api-keys
2. Create a new key
3. Add to `.env` as `OPENAI_API_KEY`

*Not required if using `mlx-whisper` (local)*

### ElevenLabs (Text-to-Speech)
1. Go to https://elevenlabs.io/
2. Create an account or sign in
3. Go to Profile → API Keys
4. Create a new key
5. Add to `.env` as `ELEVENLABS_API_KEY`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | No* | OpenAI/Whisper API key |
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs TTS key |
| `AUTH_USERNAME` | No | Basic auth username |
| `AUTH_PASSWORD` | No | Basic auth password |
| `VOICE_SERVER_URL` | No | Public URL for callbacks |

*Required only if using OpenAI Whisper instead of MLX Whisper

---

## Network & Security

### Default: Localhost Only

By default, Voxio Bot binds to `127.0.0.1` (localhost) for security. To access remotely, use a secure tunnel.

### Recommended: Cloudflare Tunnel

```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared

# Create tunnel
cloudflared tunnel create voxio-bot
cloudflared tunnel route dns voxio-bot voice.yourdomain.com

# Run tunnel
cloudflared tunnel run voxio-bot
```

### Alternative: Tailscale

```bash
# Install and connect
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up

# Access via Tailscale IP
http://<tailscale-ip>:8086/client
```

### TURN Servers

For WebRTC connections through NAT/firewalls, configure a TURN server. Recommended: [Cloudflare TURN](https://developers.cloudflare.com/calls/turn/) (free tier available).

---

## Troubleshooting

### "Module not found" errors
```bash
uv sync --reinstall
```

### Microphone not working
- Check browser permissions
- Ensure microphone is not in use by another app
- Try Chrome (recommended)

### High latency
- Enable TTS caching: `[cache] precache_fillers = true`
- Use `mlx-whisper` for local STT
- Reduce `[vad] stop_secs` to `0.2`

### Connection fails
- Add a TURN server for NAT traversal
- Check if UDP traffic is allowed
- Try disabling VPN

### Fillers not playing
- Check cache exists: `ls ~/.cache/voxio-tts/`
- Run `python precache_fillers.py` manually
- Check logs for "Preloaded X fillers"

### Handoff not working
- Ensure `clawdbot` command is in PATH
- Check Clawdbot is running
- Check logs for handoff errors

---

## Project Structure

```
voxio-bot/
├── bot.py                 # Main Pipecat bot
├── run_auth.py            # Server with auth + /speak
├── config.toml            # Configuration file
├── cached_tts.py          # TTS caching wrapper
├── filler_cache.py        # Filler pre-caching system
├── tts_cache.py           # Unified cache implementation
├── precache_fillers.py    # CLI for pre-caching
├── piper_local_tts.py     # Local Piper TTS support
├── sounds/                # Ambient sound files
│   ├── handoff_typing.wav
│   ├── office-chair.wav
│   └── ...
├── static/                # Web client assets
├── docs/
│   └── architecture.png
├── CACHING_TTS.md         # Caching documentation
├── FILLER_CACHE.md        # Filler system documentation
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── .env.example           # Environment template
└── README.md
```

---

## License

MIT
