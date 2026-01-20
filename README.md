# Vinston Wolf Voice Assistant

A real-time voice AI assistant powered by:
- **STT**: OpenAI Whisper (real-time speech recognition)
- **LLM**: Anthropic Claude (claude-sonnet-4-20250514)
- **TTS**: ElevenLabs (Roger voice - professional male)
- **Transport**: WebRTC (peer-to-peer)
- **Framework**: [Pipecat](https://pipecat.ai)

## Architecture

```
┌─────────────┐     WebRTC      ┌──────────────────────────────────────┐
│   Browser   │ ◄────────────► │           Pipecat Server              │
│  (Client)   │                 │                                      │
│             │                 │  Audio In → OpenAI Whisper STT       │
│  🎤 Mic     │    ────────►   │              ↓                        │
│  🔊 Speaker │    ◄────────   │         Claude LLM                    │
└─────────────┘                 │              ↓                        │
                                │       ElevenLabs TTS → Audio Out     │
                                └──────────────────────────────────────┘
                                        ↓               ↑
                              handoff_to_clawdbot    /speak
                                        ↓               ↑
                                ┌───────────────────────────────┐
                                │          Clawdbot             │
                                │  (Calendar, Email, Web, etc.) │
                                └───────────────────────────────┘
```

## Quick Start

### 1. Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set up environment
```bash
cd voice-assistant
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

## Clawdbot Integration

The voice assistant has bidirectional integration with Clawdbot:

### Handoff Tool (Voice → Clawdbot)

When users ask for something requiring tools (calendar, email, web search, etc.), Claude will use the `handoff_to_clawdbot` tool to delegate the task:

```
User: "What's on my calendar tomorrow?"
Claude: "I've sent that to the main system. I'll speak the results when they're ready."
→ Triggers: clawdbot wake --mode now --text "[Voice Task] Check calendar for tomorrow..."
```

### /speak Endpoint (Clawdbot → Voice)

Clawdbot can speak results back through active voice sessions:

```bash
# Check active sessions
curl https://voice.ip11.net/sessions

# Speak to the active session
curl -X POST https://voice.ip11.net/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "You have two meetings tomorrow..."}'
```

**Endpoint: POST /speak**
```json
{
  "text": "Message to speak",
  "session_id": "optional - defaults to most recent session"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123",
  "text_length": 45
}
```

**Endpoint: GET /sessions**
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
3. Clawdbot receives the task via wake command
4. Clawdbot performs web search
5. Clawdbot POSTs results to `/speak`
6. Voice assistant speaks the results to the user

## API Keys Required

### Anthropic (Claude LLM)
1. Go to https://console.anthropic.com/
2. Create an account or sign in
3. Navigate to API Keys
4. Create a new key
5. Add to `.env` as `ANTHROPIC_API_KEY`

### OpenAI (Whisper STT)
1. Go to https://platform.openai.com/api-keys
2. Create an account or sign in
3. Create a new key
4. Add to `.env` as `OPENAI_API_KEY`

### ElevenLabs (Text-to-Speech)
1. Go to https://elevenlabs.io/
2. Create an account or sign in
3. Go to Profile → API Keys
4. Create a new key
5. Add to `.env` as `ELEVENLABS_API_KEY`

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI/Whisper API key |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS key |
| `AUTH_USERNAME` | Basic auth username (default: vinston) |
| `AUTH_PASSWORD` | Basic auth password |
| `VOICE_SERVER_URL` | Public URL for callbacks (default: https://voice.ip11.net) |

### Voice Selection

The default voice is "Roger" (`CwhRBWXzGAHq8TQ4Fs17`). To change it:

1. Browse ElevenLabs voices: https://elevenlabs.io/voice-library
2. Find a voice you like and copy its ID
3. Update `voice_id` in `bot.py`

### Model Selection

Claude model can be changed in `bot.py`:
- `claude-sonnet-4-20250514` (default, balanced)
- `claude-3-5-haiku-20241022` (faster, cheaper)
- `claude-opus-4-20250514` (most capable)

### VAD Settings

Voice Activity Detection parameters in `bot.py`:
- `stop_secs`: Silence duration before considering turn complete (default: 0.3s)

## Vinston Wolf Personality

The assistant is configured as "Vinston Wolf" with these traits:
- Cool and collected
- Calm demeanor
- Professional problem-solver
- Concise responses (optimized for voice)
- Friendly but professional

Edit `VINSTON_SYSTEM_PROMPT` in `bot.py` to customize the personality.

## Troubleshooting

### "Module not found" errors
```bash
uv sync --reinstall
```

### Microphone not working
- Check browser permissions
- Ensure microphone is not in use by another app
- Try a different browser (Chrome recommended)

### High latency
- Try `eleven_turbo_v2_5` for TTS (already default)
- Reduce `stop_secs` VAD parameter
- Check your network connection

### Connection fails
- WebRTC may be blocked by VPN/firewall
- Try disabling VPN
- Check if UDP traffic is allowed

### Handoff not working
- Ensure `clawdbot` command is in PATH
- Check Clawdbot daemon is running: `clawdbot daemon status`
- Check logs: `tail -f /tmp/voice-assistant.log`

## Project Structure

```
voice-assistant/
├── bot.py              # Main Pipecat bot with Clawdbot integration
├── run_auth.py         # Server runner with auth + /speak endpoint
├── pyproject.toml      # Python dependencies
├── .env.example        # Environment template
├── .env                # Your API keys (gitignored)
└── README.md           # This file
```

## License

MIT
