#
# Voxio Bot (voxio.bot)
# Real-time voice + video AI using Pipecat + Claude + ElevenLabs
# With Clawdbot bidirectional communication support
#

"""
Voxio Bot - Voice & video AI assistant.

Run locally with:
    uv run python bot.py

With custom config:
    uv run python bot.py --config my-config.toml

LLM mode:
    uv run python bot.py --llm-mode local    # Default: VB's Claude + limited tools + handoff
    uv run python bot.py --llm-mode gateway  # Full Clawdbot access (all tools, no handoff)

Session targeting (ACP-style):
    uv run python bot.py --session agent:main:main
    uv run python bot.py --session-label "voice assistant"
    uv run python bot.py --require-existing
    uv run python bot.py --reset-session

Then open http://localhost:8086/client in your browser.
"""

import os
import io
import uuid
import signal
import asyncio
import subprocess
import tomllib
from pathlib import Path
from typing import Optional, Any

import aiohttp

from dotenv import load_dotenv
from loguru import logger

# ============================================================================
# FORCE H264 CODEC (monkey-patch aiortc to prefer H264 over VP8)
# ============================================================================
def _patch_aiortc_h264():
    """Patch aiortc to prefer H264 codec for video (better for screen sharing)."""
    try:
        from aiortc import RTCPeerConnection
        from aiortc.rtcrtpsender import RTCRtpSender
        
        _original_addTrack = RTCPeerConnection.addTrack
        
        def _patched_addTrack(self, track, *streams):
            sender = _original_addTrack(self, track, *streams)
            
            # Set H264 as preferred codec for video tracks
            if track.kind == 'video':
                try:
                    caps = RTCRtpSender.getCapabilities('video')
                    if caps and caps.codecs:
                        # Sort codecs to put H264 first
                        h264_codecs = [c for c in caps.codecs if 'H264' in c.mimeType]
                        other_codecs = [c for c in caps.codecs if 'H264' not in c.mimeType]
                        sorted_codecs = h264_codecs + other_codecs
                        
                        # Find and configure the transceiver
                        for t in self.getTransceivers():
                            if t.sender == sender:
                                t.setCodecPreferences(sorted_codecs)
                                logger.info("🎬 Set H264 as preferred video codec")
                                break
                except Exception as e:
                    logger.warning(f"Could not set H264 preference: {e}")
            
            return sender
        
        RTCPeerConnection.addTrack = _patched_addTrack
        logger.info("✅ Patched aiortc to prefer H264 codec")
    except Exception as e:
        logger.warning(f"Could not patch aiortc for H264: {e}")

_patch_aiortc_h264()


# ============================================================================
# TOML CONFIGURATION
# ============================================================================

def _get_config_path() -> Path:
    """Get config path from --config CLI arg or default to config.toml."""
    import sys
    for i, arg in enumerate(sys.argv):
        if arg in ('--config', '-c') and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return Path(__file__).parent / "config.toml"

def _get_cli_arg(names: list, default=None):
    """Get a CLI argument value by name(s)."""
    import sys
    for i, arg in enumerate(sys.argv):
        if arg in names and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

def _has_cli_flag(names: list) -> bool:
    """Check if a CLI flag is present."""
    import sys
    return any(arg in sys.argv for arg in names)

CONFIG_PATH = _get_config_path()
_config: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    """Load configuration from TOML config file."""
    global _config
    try:
        with open(CONFIG_PATH, "rb") as f:
            _config = tomllib.load(f)
        logger.info(f"📋 Loaded config from {CONFIG_PATH}")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {CONFIG_PATH}, using defaults")
        _config = {}
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Invalid TOML in {CONFIG_PATH}: {e}")
        _config = {}
    return _config


def get_config(key: str, default: Any = None) -> Any:
    """Get a config value by dot-notation key (e.g., 'handoff.type')."""
    keys = key.split(".")
    value = _config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
        if value is None:
            return default
    return value


# Load config at startup
load_config()

print("🎙️ Starting Voxio Bot...")
print("⏳ Loading models and imports (may take ~20 seconds on first run)\n")

# Load VAD for voice activity detection
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TextFrame, TTSSpeakFrame, InputImageRawFrame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext, ToolsSchema
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import LLMService
from pipecat.frames.frames import TextFrame, LLMFullResponseEndFrame


class GatewayLLMService(LLMService):
    """
    LLM Service that routes all requests through Clawdbot Gateway.
    Provides full tool access without handoff - voice becomes a direct interface.
    """
    
    def __init__(self, session_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._session_key = session_key or "agent:main:voice"
        self._callback_url = "http://localhost:8086/speak"
        logger.info(f"🌐 GatewayLLMService initialized (session: {self._session_key})")
    
    async def process_frame(self, frame, direction):
        """Process frames - handle LLMContextFrame to trigger gateway."""
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.frame_processor import FrameDirection
        
        # Handle context frames - this is what triggers the LLM
        if isinstance(frame, LLMContextFrame):
            logger.info("🌐 Received LLMContextFrame, processing...")
            await self._process_context(frame)
            # Don't call super - we handle this frame ourselves
            return
        
        # Pass other frames through normally
        await super().process_frame(frame, direction)
    
    async def _process_context(self, frame):
        """Process the conversation context through Clawdbot Gateway."""
        from pipecat.frames.frames import LLMFullResponseStartFrame
        
        await self.push_frame(LLMFullResponseStartFrame())
        
        # Get messages from the frame's context or directly from frame
        context = getattr(frame, 'context', None)
        if context and hasattr(context, 'get_messages'):
            messages = context.get_messages()
        elif hasattr(frame, 'messages'):
            messages = frame.messages
        else:
            messages = []
        
        # Find the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multi-part content (text + images)
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    content = " ".join(text_parts)
                user_message = content
                break
        
        if not user_message:
            logger.warning("🌐 No user message found in context")
            await self.push_frame(LLMFullResponseEndFrame())
            return
        
        logger.info(f"🌐 Gateway request: {user_message[:100]}...")
        
        # Direct response mode - no callback, speak the response directly
        full_message = f"[Voice Task - respond concisely in 1-2 sentences] {user_message}"
        
        # Start ambient sounds while waiting for gateway
        await audio_manager.start_ambient()
        
        try:
            # Call clawdbot agent with the session
            proc = await asyncio.create_subprocess_exec(
                "clawdbot", "agent",
                "--session-id", self._session_key,
                "--message", full_message,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            await audio_manager.stop_ambient()
            
            if proc.returncode == 0:
                response = stdout.decode().strip()
                logger.info(f"🌐 Gateway raw response: {response[:300]}...")
                
                # Extract text from /speak tool call if present
                # Look for: "text": "actual response"
                import re
                match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
                if match:
                    speak_text = match.group(1)
                    # Unescape JSON
                    speak_text = speak_text.replace('\\n', '\n').replace('\\"', '"')
                    logger.info(f"🌐 Extracted speak text: {speak_text[:100]}...")
                    await self.push_frame(TextFrame(speak_text))
                elif response and not response.startswith('exec:'):
                    await self.push_frame(TextFrame(response))
                else:
                    await self.push_frame(TextFrame("I processed your request."))
            else:
                error = stderr.decode().strip()
                logger.error(f"🌐 Gateway error: {error}")
                await self.push_frame(TextFrame("I had trouble reaching my backend. Please try again."))
                
        except asyncio.TimeoutError:
            await audio_manager.stop_ambient()
            logger.error("🌐 Gateway timeout")
            await self.push_frame(TextFrame("The request timed out. Please try again."))
        except Exception as e:
            await audio_manager.stop_ambient()
            logger.error(f"🌐 Gateway exception: {e}")
            await self.push_frame(TextFrame("Something went wrong. Please try again."))
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

# Conditional imports for local Whisper
try:
    from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTServiceMLX, Model as WhisperModel, MLXModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Local Whisper not available. Install with: uv add pipecat-ai[whisper] or pipecat-ai[mlx-whisper]")

# Conditional imports for local Moondream vision
MOONDREAM_AVAILABLE = False
_moondream_model = None
_moondream_device = None

def _load_moondream_model():
    """Load Moondream model. Called at startup if vision.mode = 'local'."""
    global _moondream_model, _moondream_device, MOONDREAM_AVAILABLE
    
    if _moondream_model is not None:
        return _moondream_model
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        # Detect best device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        
        logger.info(f"🌙 Loading Moondream model on {device}...")
        
        _moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            revision="2025-01-09",
            device_map={"": device},
            dtype=dtype,
        ).eval()
        
        _moondream_device = device
        MOONDREAM_AVAILABLE = True
        logger.info("✅ Moondream model loaded")
        return _moondream_model
        
    except Exception as e:
        logger.error(f"Failed to load Moondream: {e}")
        return None

def _get_moondream_model():
    """Get loaded Moondream model (or load if not yet loaded)."""
    if _moondream_model is None:
        return _load_moondream_model()
    return _moondream_model
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# ============================================================================
# CONFIGURATION (from config.toml, with .env fallback for secrets)
# ============================================================================

# LLM configuration
def get_llm_mode():
    """Get LLM mode: 'local' or 'gateway' (--llm-mode)."""
    cli_val = _get_cli_arg(['--llm-mode'])
    return cli_val if cli_val else get_config("llm.mode", "local")


# Session configuration (ACP-style options - CLI overrides config)
def get_session_key():
    """Get session key for Clawdbot handoffs (--session <key>)."""
    cli_val = _get_cli_arg(['--session', '-s'])
    return cli_val if cli_val else get_config("session.key", None)

def get_session_label():
    """Get session label to resolve (--session-label <label>)."""
    cli_val = _get_cli_arg(['--session-label'])
    return cli_val if cli_val else get_config("session.label", None)

def get_session_require_existing():
    """Whether to require session to already exist (--require-existing)."""
    if _has_cli_flag(['--require-existing']):
        return True
    return get_config("session.require_existing", False)

def get_session_reset_on_connect():
    """Whether to reset session on connect (--reset-session)."""
    if _has_cli_flag(['--reset-session']):
        return True
    return get_config("session.reset_on_connect", False)


def get_identity_name():
    """Get bot identity name from config."""
    return get_config("identity.name", "Vinston Wolf")

def get_random_greeting():
    """Get a random greeting from the config list."""
    import random
    greetings = get_config("identity.greetings", ["Vinston here. What needs fixing?"])
    if isinstance(greetings, list) and greetings:
        return random.choice(greetings)
    return greetings if greetings else "Hello!"

def get_stt_provider():
    return get_config("stt.provider", os.getenv("STT_PROVIDER", "openai")).lower()

def get_stt_model():
    return get_config("stt.model", os.getenv("STT_MODEL", "base")).lower()

def get_tts_provider():
    return get_config("tts.provider", os.getenv("TTS_PROVIDER", "elevenlabs")).lower()

def get_tts_voice():
    return get_config("tts.voice", os.getenv("TTS_VOICE", "Roger"))

def get_tts_model_path():
    return os.getenv("TTS_MODEL_PATH", "")

def get_video_in_enabled():
    return get_config("video.input_enabled", os.getenv("VIDEO_IN_ENABLED", "false").lower() == "true")

def get_video_out_enabled():
    return get_config("video.output_enabled", os.getenv("VIDEO_OUT_ENABLED", "false").lower() == "true")

def get_default_avatar_path():
    return get_config("video.avatar", os.getenv("DEFAULT_AVATAR_PATH", "avatar.png"))

def get_image_display_duration():
    """Get how long to display generated images before returning to avatar (5 min default)."""
    return get_config("video.image_display_duration", 300)

def get_capture_display_duration():
    """Get how long to display captured frames before returning to avatar (60s default)."""
    return get_config("video.capture_display_duration", 60)

def get_gif_display_duration():
    """Get how long to display GIFs before returning to avatar (2 min default)."""
    return get_config("video.gif_display_duration", 120)

async def cancel_current_display():
    """Cancel any current image display and return to avatar."""
    global _avatar_return_task, _gif_animation_task, _image_display_until
    
    # Cancel avatar return timer
    if _avatar_return_task and not _avatar_return_task.done():
        _avatar_return_task.cancel()
        logger.info("🖼️ Cancelled avatar return timer")
    
    # Cancel GIF animation
    if _gif_animation_task and not _gif_animation_task.done():
        _gif_animation_task.cancel()
        logger.info("🖼️ Cancelled GIF animation")
    
    # Reset display timer
    _image_display_until = 0
    
    # Return to avatar
    await send_default_avatar()
    logger.info("🖼️ Returned to avatar for new image")

def get_vision_mode():
    """Get vision analysis mode: handoff, direct, or local."""
    return get_config("vision.mode", "handoff")

def get_vision_local_model():
    """Get local vision model name."""
    return get_config("vision.local_model", "moondream")

def get_vision_local_model_path():
    """Get local vision model path."""
    return get_config("vision.local_model_path", None)

def get_voice_server_url():
    return get_config("server.url", os.getenv("VOICE_SERVER_URL", "http://localhost:8086"))

def get_vad_stop_secs():
    return get_config("vad.stop_secs", 0.3)

# Legacy globals for compatibility (will be removed later)
STT_PROVIDER = get_stt_provider()
STT_MODEL = get_stt_model()
TTS_PROVIDER = get_tts_provider()
TTS_VOICE = get_tts_voice()
TTS_MODEL_PATH = get_tts_model_path()
VIDEO_IN_ENABLED = get_video_in_enabled()
VIDEO_OUT_ENABLED = get_video_out_enabled()
DEFAULT_AVATAR_PATH = get_default_avatar_path()

# Eager-load Moondream if vision mode is "local"
if get_vision_mode() == "local":
    logger.info("🔭 Vision mode is 'local' - loading Moondream at startup...")
    _load_moondream_model()

# Nano-banana path for image generation
NANO_BANANA_SCRIPT = "/opt/homebrew/lib/node_modules/clawdbot/skills/nano-banana-pro/scripts/generate_image.py"


def create_stt_service():
    """Create the appropriate STT service based on configuration."""
    provider = STT_PROVIDER
    model = STT_MODEL
    
    if provider == "openai":
        logger.info("🎤 Using OpenAI Whisper API (cloud)")
        return OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    
    elif provider == "mlx-whisper" or provider == "mlx":
        if not WHISPER_AVAILABLE:
            logger.error("MLX Whisper not installed. Run: uv add pipecat-ai[mlx-whisper]")
            raise RuntimeError("MLX Whisper not available")
        
        # Map model names to MLXModel enum
        mlx_models = {
            "tiny": MLXModel.TINY,
            "medium": MLXModel.MEDIUM,
            "large": MLXModel.LARGE_V3,
            "large-turbo": MLXModel.LARGE_V3_TURBO,
            "large-turbo-q4": MLXModel.LARGE_V3_TURBO_Q4,
            "distil-large": MLXModel.DISTIL_LARGE_V3,
        }
        selected_model = mlx_models.get(model, MLXModel.LARGE_V3_TURBO)
        logger.info(f"🎤 Using MLX Whisper (local Apple Silicon) - model: {selected_model.value}")
        return WhisperSTTServiceMLX(model=selected_model)
    
    elif provider == "whisper" or provider == "faster-whisper":
        if not WHISPER_AVAILABLE:
            logger.error("Faster Whisper not installed. Run: uv add pipecat-ai[whisper]")
            raise RuntimeError("Faster Whisper not available")
        
        # Map model names to WhisperModel enum
        whisper_models = {
            "tiny": WhisperModel.TINY,
            "base": WhisperModel.BASE,
            "small": WhisperModel.SMALL,
            "medium": WhisperModel.MEDIUM,
            "large": WhisperModel.LARGE,
            "large-turbo": WhisperModel.LARGE_V3_TURBO,
            "distil-large": WhisperModel.DISTIL_LARGE_V2,
            "distil-medium-en": WhisperModel.DISTIL_MEDIUM_EN,
        }
        selected_model = whisper_models.get(model, WhisperModel.BASE)
        logger.info(f"🎤 Using Faster Whisper (local) - model: {selected_model.value}")
        return WhisperSTTService(model=selected_model)
    
    else:
        logger.warning(f"Unknown STT provider '{provider}', falling back to OpenAI")
        return OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))


def create_tts_service():
    """Create the appropriate TTS service based on configuration."""
    provider = TTS_PROVIDER
    voice = TTS_VOICE
    model_path = TTS_MODEL_PATH
    
    if provider == "elevenlabs":
        # ElevenLabs voice IDs - map friendly names to IDs
        voice_ids = {
            "roger": "CwhRBWXzGAHq8TQ4Fs17",
            "adam": "pNInz6obpgDQGcFmaJgB",
            "josh": "TxGEqnHWrfWFTfGW9XjX",
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "bella": "EXAVITQu4vr4xnSDxMaL",
        }
        voice_id = voice_ids.get(voice.lower(), voice)  # Use as-is if not in map
        
        logger.info(f"🔊 Using ElevenLabs TTS (cloud) - voice: {voice}")
        return ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=voice_id,
            model="eleven_turbo_v2_5",
        )
    
    elif provider == "piper":
        # Local Piper TTS
        from piper_local_tts import PiperLocalTTSService
        
        # Default to ryan-high if no model path specified
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "piper_voices",
                "en_US-ryan-high.onnx"
            )
        
        logger.info(f"🔊 Using Piper TTS (local) - model: {os.path.basename(model_path)}")
        return PiperLocalTTSService(model_path=model_path)
    
    else:
        logger.warning(f"Unknown TTS provider '{provider}', falling back to ElevenLabs")
        return ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="CwhRBWXzGAHq8TQ4Fs17",
            model="eleven_turbo_v2_5",
        )

# ============================================================================
# PATCH PIPECAT APP TO ADD /speak ENDPOINT
# ============================================================================
import pipecat.runner.run as pipecat_runner

_original_create_server_app = pipecat_runner._create_server_app

def _patched_create_server_app(**kwargs):
    """Add /speak endpoint to pipecat's FastAPI app."""
    app = _original_create_server_app(**kwargs)
    
    from fastapi import Request
    from fastapi.responses import JSONResponse
    
    @app.post("/speak")
    async def speak_endpoint(request: Request):
        """Inject text into an active voice session for TTS."""
        try:
            body = await request.json()
            text = body.get("text")
            session_id = body.get("session_id")
            
            if not text:
                return JSONResponse(
                    content={"success": False, "error": "Missing 'text' field"},
                    status_code=400
                )
            
            # Interrupt all audio before speaking (stop ambient + current TTS)
            await audio_manager.interrupt_all()
            logger.info("🔇 /speak: Audio interrupted for injection")
            
            from sessions import speak_to_session
            result = await speak_to_session(text, session_id)
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"/speak error: {e}")
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500
            )
    
    @app.get("/sessions")
    async def sessions_endpoint():
        """Get info about active sessions."""
        from sessions import get_session_info
        return JSONResponse(content=get_session_info())
    
    logger.info("✅ Added /speak and /sessions endpoints")
    return app

pipecat_runner._create_server_app = _patched_create_server_app

# ============================================================================
# ACTIVE SESSION TRACKING - imported from sessions.py
# ============================================================================
from sessions import (
    register_session as _register,
    unregister_session as _unregister,
    active_sessions,
    speak_to_session,
    get_session_info,
)


def register_session(session_id: str, task: PipelineTask, tts, context_messages: list):
    """Register an active voice session."""
    _register(session_id, task, tts, context_messages)


def unregister_session(session_id: str):
    """Remove a session when client disconnects."""
    _unregister(session_id)


# ============================================================================
# CLAWDBOT HANDOFF TOOL
# ============================================================================

VOICE_SERVER_URL = os.getenv("VOICE_SERVER_URL", "https://voice.ip11.net")


async def speak_then_play_handoff_sound():
    """Speak acknowledgment, then play handoff sound.
    
    Order: 1) Speak brief acknowledgment via TTS, 2) Play typing sound
    """
    global _current_task
    
    if not _current_task:
        return
    
    # Speak acknowledgment first
    acknowledgments = [
        "Let me check on that.",
        "One moment.",
        "Let me look into that.",
        "Checking now.",
    ]
    import random
    ack = random.choice(acknowledgments)
    
    try:
        # Queue the spoken acknowledgment
        await _current_task.queue_frame(TTSSpeakFrame(text=ack))
        logger.info(f"🗣️ Spoke acknowledgment: {ack}")
        
        # Small delay to let TTS finish before typing sound
        await asyncio.sleep(1.5)
        
        # Then play the handoff sound
        await play_handoff_sound()
    except Exception as e:
        logger.warning(f"Could not speak acknowledgment: {e}")


# ============================================================================
# AMBIENT SOUND LOOP - plays random sounds while waiting for Clawdbot
# ============================================================================

def get_handoff_type():
    """Get handoff sound type from config."""
    return get_config("handoff.type", "ambient").lower()

def get_handoff_prompt():
    """Get handoff sound generation prompt from config."""
    return get_config("handoff.prompt", "soft keyboard typing")

def get_handoff_files():
    """Get list of handoff sound files from config."""
    default_files = [
        "sounds/handoff_typing.wav",
        "sounds/110451__freeborn__paper01.wav",
        "sounds/377260__johnnypanic__clearing-throat-2.wav",
        "sounds/414819__bokal__office-drawer.wav",
    ]
    return get_config("handoff.files", default_files)

def get_handoff_gap():
    """Get gap between sounds from config."""
    return get_config("handoff.gap", 0.5)

# Global task reference for the ambient sound loop
_ambient_sound_task: Optional[asyncio.Task] = None
_ambient_sounds_playing: bool = False


def stop_ambient_sounds():  # Legacy - use audio_manager.stop_ambient()
    """Stop the ambient sound loop if running and interrupt audio playback."""
    global _ambient_sound_task, _ambient_sounds_playing, _current_task
    
    if _ambient_sound_task and not _ambient_sound_task.done():
        _ambient_sound_task.cancel()
        logger.info("🔇 Ambient sounds stopped")
    
    _ambient_sound_task = None
    _ambient_sounds_playing = False
    
    # Send interruption frame to stop audio playback
    if _current_task:
        try:
            import asyncio
            asyncio.create_task(_send_interruption_frame())
        except Exception as e:
            logger.warning(f"Could not send interruption frame: {e}")


async def _send_interruption_frame():
    """Send an interruption frame to stop current audio playback."""
    global _current_task
    if _current_task:
        from pipecat.frames.frames import InterruptionTaskFrame
        # Push frame (Pipecat 0.0.99 API - no direction argument)
        await _current_task.queue_frame(InterruptionTaskFrame())
        logger.info("🔇 Sent InterruptionTaskFrame to stop audio")


def is_ambient_sounds_playing() -> bool:
    """Check if ambient sounds are currently playing."""
    return _ambient_sounds_playing


async def _play_single_sound(sound_file: str) -> float:
    """Play a single sound file and return its duration in seconds."""
    global _current_task
    
    if not _current_task:
        return 0
    
    from pipecat.frames.frames import OutputAudioRawFrame
    import wave
    
    file_path = os.path.join(os.path.dirname(__file__), sound_file)
    
    if not os.path.exists(file_path):
        logger.warning(f"Sound file not found: {file_path}")
        return 0
    
    try:
        # Read WAV file properties to get duration
        with wave.open(file_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / sample_rate
        
        # Read raw audio data (skip 44-byte header)
        with open(file_path, 'rb') as f:
            wav_data = f.read()
            audio_data = wav_data[44:]
        
        frame = OutputAudioRawFrame(
            audio=audio_data,
            sample_rate=sample_rate,
            num_channels=1
        )
        await _current_task.queue_frame(frame)
        logger.info(f"🔊 Playing: {os.path.basename(sound_file)} ({duration:.1f}s)")
        
        return duration
        
    except Exception as e:
        logger.warning(f"Could not play sound {sound_file}: {e}")
        return 0


async def _ambient_sound_loop():
    """Loop that plays random ambient sounds until interrupted."""
    global _ambient_sounds_playing
    
    import random
    
    _ambient_sounds_playing = True
    files = get_handoff_files()
    gap = get_handoff_gap()
    
    logger.info(f"🎵 Starting ambient sound loop ({len(files)} files, {gap}s gap)")
    
    try:
        while _ambient_sounds_playing:
            # Pick a random sound
            sound_file = random.choice(files)
            
            # Play it and get duration
            duration = await _play_single_sound(sound_file)
            
            if duration > 0:
                # Wait for the sound to finish playing, plus configured gap
                await asyncio.sleep(duration + gap)
            else:
                # If sound failed, wait a bit before trying another
                await asyncio.sleep(1.0)
                
    except asyncio.CancelledError:
        logger.info("🎵 Ambient sound loop cancelled")
    except Exception as e:
        logger.error(f"Ambient sound loop error: {e}")
    finally:
        _ambient_sounds_playing = False


def start_ambient_sounds():
    """Start the ambient sound loop in the background."""
    global _ambient_sound_task
    
    # Stop any existing loop first
    stop_ambient_sounds()
    
    # Start new loop
    _ambient_sound_task = asyncio.create_task(_ambient_sound_loop())
    logger.info("🎵 Ambient sound loop started")


async def play_handoff_sound():
    """Play sound while waiting for handoff response.
    
    Behavior depends on handoff.type in config.toml:
    - none: no sound
    - prompt: generate via ElevenLabs SFX API
    - ambient: loop random files until interrupted
    """
    global _current_task
    
    sound_type = get_handoff_type()
    
    if sound_type == "none" or not sound_type:
        logger.info("🔇 Handoff sound disabled (type=none)")
        return
    
    if not _current_task:
        return
    
    from pipecat.frames.frames import OutputAudioRawFrame
    
    if sound_type == "prompt":
        # Generate via ElevenLabs SFX API
        sound_prompt = get_handoff_prompt()
        api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not api_key:
            logger.warning("No ElevenLabs API key for handoff sound generation")
            return
        
        logger.info(f"🔊 Generating handoff sound: {sound_prompt[:30]}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.elevenlabs.io/v1/sound-generation",
                    headers={
                        "xi-api-key": api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": sound_prompt,
                        "duration_seconds": 10.0,
                    },
                    params={
                        "output_format": "pcm_24000" if get_tts_provider() != "piper" else "pcm_22050"
                    }
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        
                        sfx_sample_rate = 22050 if get_tts_provider() == "piper" else 24000
                        frame = OutputAudioRawFrame(
                            audio=audio_data,
                            sample_rate=sfx_sample_rate,
                            num_channels=1
                        )
                        await _current_task.queue_frame(frame)
                        logger.info("🔊 Handoff sound (generated) queued")
                    else:
                        logger.warning(f"Handoff sound generation failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Could not generate handoff sound: {e}")
            
    elif sound_type == "ambient":
        # Start the ambient sound loop
        asyncio.create_task(audio_manager.start_ambient())
        
    else:
        logger.warning(f"Unknown handoff sound type: {sound_type}")


async def handoff_to_clawdbot(params) -> dict:
    """
    Hand off a task to Clawdbot for execution.
    
    This function is called when Claude needs to perform actions that require
    Clawdbot's tools (calendar, email, web search, file operations, etc.)
    
    Args:
        params: FunctionCallParams containing the task in arguments
        
    Returns:
        Status message
    """
    # Extract task from FunctionCallParams
    task = params.arguments.get("task", "")
    # Include callback URL so Clawdbot can speak results back
    callback_info = f"[Callback: POST {VOICE_SERVER_URL}/speak with JSON {{\"text\": \"your response\"}}]"
    full_message = f"[Voice Task] {task}\n\n{callback_info}"
    
    try:
        # Build clawdbot command with session options
        session_key = get_session_key()
        session_label = get_session_label()
        
        # Log session targeting
        if session_key:
            logger.info(f"🎯 Session key: {session_key}")
        if session_label:
            logger.info(f"🏷️ Session label: {session_label}")
        
        # Use 'agent' command if session targeting is configured, else 'wake'
        if session_key:
            # Use agent command with explicit session
            cmd = ["clawdbot", "agent", "--session-id", session_key, "--message", full_message]
            logger.info(f"🤖 Handing off to Clawdbot (session: {session_key}): {task[:100]}...")
        else:
            # Default: use wake command (routes to main session)
            cmd = ["clawdbot", "wake", "--mode", "now", "--text", full_message]
            logger.info(f"🤖 Handing off to Clawdbot: {task[:100]}...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            logger.info("✅ Clawdbot wake successful")
            # Start ambient sounds directly (no spoken acknowledgment)
            await audio_manager.start_ambient()
            # Return empty message - Clawdbot will respond via /speak
            return {"status": "success", "message": ""}
        else:
            logger.error(f"Clawdbot wake failed: {result.stderr}")
            return {"status": "error", "message": f"I couldn't reach the main system. {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        logger.error("Clawdbot wake timed out")
        return {"status": "pending", "message": "Let me check with my manager. This might take a moment."}
    except FileNotFoundError:
        logger.error("clawdbot command not found")
        return {"status": "error", "message": "I'm sorry, the main system is not available right now."}
    except Exception as e:
        logger.error(f"Handoff error: {e}")
        return f"Error during handoff: {str(e)}"


# ============================================================================
# SOUND EFFECTS TOOL
# ============================================================================

# Global reference to task for sound effect injection
_current_task: Optional["PipelineTask"] = None


async def play_sound_effect(params) -> dict:
    """
    Generate and play a sound effect using ElevenLabs Sound Generation API.
    
    Args:
        params: FunctionCallParams containing the sound description
        
    Returns:
        Status message
    """
    from pipecat.frames.frames import OutputAudioRawFrame
    
    prompt = params.arguments.get("prompt", "")
    duration = params.arguments.get("duration", 2.0)  # Default 2 seconds
    
    if not prompt:
        return {"status": "error", "message": "No sound effect description provided"}
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return {"status": "error", "message": "ElevenLabs API key not configured"}
    
    try:
        logger.info(f"🔊 Generating sound effect: {prompt[:50]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.elevenlabs.io/v1/sound-generation",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": prompt,
                    "duration_seconds": min(duration, 5.0),  # Cap at 5 seconds
                },
                params={
                    # Match TTS sample rate: 22050 for Piper, 24000 for ElevenLabs
                    "output_format": "pcm_22050" if TTS_PROVIDER == "piper" else "pcm_24000"
                }
            ) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    logger.info(f"✅ Sound effect generated: {len(audio_data)} bytes")
                    
                    # Inject audio into the pipeline if we have a task reference
                    if _current_task:
                        # Create an audio frame with the PCM data
                        # Match TTS sample rate
                        sfx_sample_rate = 22050 if TTS_PROVIDER == "piper" else 24000
                        frame = OutputAudioRawFrame(
                            audio=audio_data,
                            sample_rate=sfx_sample_rate,
                            num_channels=1
                        )
                        await _current_task.queue_frame(frame)
                        logger.info("🔊 Sound effect queued to pipeline")
                        return {"status": "success", "message": "Sound effect playing"}
                    else:
                        logger.warning("No active task to play sound effect")
                        return {"status": "error", "message": "No active audio session"}
                else:
                    error_text = await resp.text()
                    logger.error(f"Sound effect API error: {resp.status} - {error_text}")
                    return {"status": "error", "message": f"API error: {resp.status}"}
                    
    except Exception as e:
        logger.error(f"Sound effect error: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# VIDEO/IMAGE TOOLS
# ============================================================================

from enum import Enum, auto

class DisplayState(Enum):
    """States for the video output display."""
    AVATAR = auto()     # Showing default avatar
    CAPTURE = auto()    # Showing captured camera frame
    GIF = auto()        # Animating a GIF
    IMAGE = auto()      # Showing generated image


class DisplayManager:
    """
    State machine for managing video output display.
    
    Handles transitions between avatar, captured frames, GIFs, and generated images.
    Ensures clean cancellation of timers and animations when switching.
    """
    
    def __init__(self):
        self.state = DisplayState.AVATAR
        self._return_task: Optional[asyncio.Task] = None
        self._animation_task: Optional[asyncio.Task] = None
        self._task: Optional["PipelineTask"] = None  # Reference to Pipecat task
        
    def set_task(self, task):
        """Set the Pipecat task reference for sending frames."""
        self._task = task
        
    async def _cancel_current(self):
        """Cancel any running timers or animations."""
        if self._return_task and not self._return_task.done():
            self._return_task.cancel()
            try:
                await self._return_task
            except asyncio.CancelledError:
                pass
            self._return_task = None
            
        if self._animation_task and not self._animation_task.done():
            self._animation_task.cancel()
            try:
                await self._animation_task
            except asyncio.CancelledError:
                pass
            self._animation_task = None
    
    async def _send_frame(self, image_bytes: bytes, size: tuple, format: str = "RGB"):
        """Send a frame to video output."""
        if not self._task:
            logger.warning("DisplayManager: No task set, cannot send frame")
            return False
        try:
            from pipecat.frames.frames import OutputImageRawFrame
            frame = OutputImageRawFrame(image=image_bytes, size=size, format=format)
            await self._task.queue_frame(frame)
            return True
        except Exception as e:
            logger.error(f"DisplayManager: Failed to send frame: {e}")
            return False
    
    async def _send_avatar(self):
        """Send the default avatar."""
        avatar_path = get_default_avatar_path()
        if not os.path.exists(avatar_path):
            logger.warning(f"DisplayManager: Avatar not found: {avatar_path}")
            return
        try:
            from PIL import Image
            with Image.open(avatar_path) as img:
                img_rgb = img.convert('RGB')
                await self._send_frame(img_rgb.tobytes(), img_rgb.size, "RGB")
            logger.info("🐺 DisplayManager: Sent avatar")
        except Exception as e:
            logger.error(f"DisplayManager: Failed to send avatar: {e}")
    
    async def _schedule_return(self, duration: int):
        """Schedule return to avatar after duration seconds."""
        async def return_to_avatar():
            await asyncio.sleep(duration)
            await self.show_avatar()
        
        self._return_task = asyncio.create_task(return_to_avatar())
    
    async def show_avatar(self):
        """Transition to AVATAR state."""
        import traceback
        caller = ''.join(traceback.format_stack()[-3:-1])
        logger.info(f"🖼️ DisplayManager: show_avatar() called from:\n{caller}")
        await self._cancel_current()
        self.state = DisplayState.AVATAR
        await self._send_avatar()
        logger.info("🖼️ DisplayManager: State -> AVATAR")
    
    async def show_capture(self, frame_bytes: bytes):
        """Show a captured camera frame."""
        await self._cancel_current()
        await self._send_avatar()  # Brief avatar while preparing
        
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(frame_bytes)) as img:
                img_rgb = img.convert('RGB')
                await self._send_frame(img_rgb.tobytes(), img_rgb.size, "RGB")
            
            self.state = DisplayState.CAPTURE
            duration = get_capture_display_duration()
            await self._schedule_return(duration)
            logger.info(f"🖼️ DisplayManager: State -> CAPTURE ({duration}s)")
        except Exception as e:
            logger.error(f"DisplayManager: Failed to show capture: {e}")
            await self._send_avatar()
    
    async def show_image(self, image_path: str):
        """Show a generated image."""
        await self._cancel_current()
        await self._send_avatar()  # Brief avatar while preparing
        
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')
                await self._send_frame(img_rgb.tobytes(), img_rgb.size, "RGB")
            
            self.state = DisplayState.IMAGE
            duration = get_image_display_duration()
            await self._schedule_return(duration)
            logger.info(f"🖼️ DisplayManager: State -> IMAGE ({duration}s)")
        except Exception as e:
            logger.error(f"DisplayManager: Failed to show image: {e}")
            await self._send_avatar()
    
    async def show_gif(self, frames: list, durations: list):
        """Animate a GIF."""
        await self._cancel_current()
        await self._send_avatar()  # Brief avatar while preparing
        
        self.state = DisplayState.GIF
        duration = get_gif_display_duration()
        
        async def animate():
            try:
                end_time = asyncio.get_event_loop().time() + duration
                frame_idx = 0
                
                while asyncio.get_event_loop().time() < end_time:
                    frame = frames[frame_idx % len(frames)]
                    frame_duration = durations[frame_idx % len(durations)]
                    
                    await self._send_frame(frame.tobytes(), frame.size, "RGB")
                    await asyncio.sleep(frame_duration)
                    frame_idx += 1
                
                logger.info("🎬 DisplayManager: GIF animation complete")
            except asyncio.CancelledError:
                logger.info("🎬 DisplayManager: GIF animation cancelled")
            finally:
                await self.show_avatar()
        
        self._animation_task = asyncio.create_task(animate())
        logger.info(f"🖼️ DisplayManager: State -> GIF ({duration}s, {len(frames)} frames)")


# Global display manager instance
display_manager = DisplayManager()


class AudioState(Enum):
    """States for audio output management."""
    IDLE = auto()       # Nothing playing
    TTS = auto()        # LLM speaking (interruptible)
    AMBIENT = auto()    # Background sounds (while waiting)
    INJECTED = auto()   # External /speak TTS
    SFX = auto()        # Sound effect (short, overlays)


class AudioManager:
    """
    State machine for managing audio output.
    
    Handles TTS, ambient sounds, /speak injection, and sound effects.
    Ensures proper interruption and state transitions.
    """
    
    def __init__(self):
        self.state = AudioState.IDLE
        self._task: Optional["PipelineTask"] = None
        self._ambient_task: Optional[asyncio.Task] = None
        self._tts_service = None  # Reference to TTS service
        self._last_ambient_file: Optional[str] = None  # Track last played to avoid repeats
        
    def set_task(self, task):
        """Set the Pipecat task reference."""
        self._task = task
        
    def set_tts(self, tts):
        """Set the TTS service reference."""
        self._tts_service = tts
    
    async def _send_interruption(self):
        """Send interruption frame to stop current audio."""
        if self._task:
            try:
                from pipecat.frames.frames import InterruptionTaskFrame
                await self._task.queue_frame(InterruptionTaskFrame())
                logger.debug("🔇 AudioManager: Sent interruption frame")
            except Exception as e:
                logger.warning(f"AudioManager: Could not send interruption: {e}")
    
    async def _cancel_ambient(self):
        """Cancel ambient sound loop if running."""
        if self._ambient_task and not self._ambient_task.done():
            self._ambient_task.cancel()
            try:
                await self._ambient_task
            except asyncio.CancelledError:
                pass
            self._ambient_task = None
            logger.info("🔇 AudioManager: Ambient sounds cancelled")
    
    async def interrupt_all(self):
        """Stop all audio (called on user speech)."""
        await self._cancel_ambient()
        await self._send_interruption()
        self.state = AudioState.IDLE
        logger.info("🔇 AudioManager: All audio interrupted -> IDLE")
    
    async def start_ambient(self):
        """Start ambient background sounds."""
        # Cancel any existing ambient
        await self._cancel_ambient()
        
        # Don't start if already doing TTS
        if self.state == AudioState.TTS:
            logger.debug("AudioManager: Skipping ambient, TTS in progress")
            return
        
        self.state = AudioState.AMBIENT
        
        async def ambient_loop():
            """Loop playing random ambient sounds."""
            import random
            files = get_handoff_files()
            gap = get_handoff_gap()
            
            logger.info(f"🎵 AudioManager: Starting ambient ({len(files)} files)")
            
            try:
                while self.state == AudioState.AMBIENT:
                    if not files:
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Pick random file, avoiding last played (if more than 1 file)
                    available = [f for f in files if f != self._last_ambient_file] if len(files) > 1 else files
                    sound_file = random.choice(available)
                    self._last_ambient_file = sound_file
                    
                    duration = await self._play_sound(sound_file)
                    
                    if duration > 0:
                        await asyncio.sleep(duration + gap)
                    else:
                        await asyncio.sleep(0.5)
                        
            except asyncio.CancelledError:
                pass
            finally:
                if self.state == AudioState.AMBIENT:
                    self.state = AudioState.IDLE
        
        self._ambient_task = asyncio.create_task(ambient_loop())
        logger.info("🎵 AudioManager: State -> AMBIENT")
    
    async def stop_ambient(self):
        """Stop ambient sounds."""
        if self.state == AudioState.AMBIENT:
            await self._cancel_ambient()
            self.state = AudioState.IDLE
            logger.info("🔇 AudioManager: State -> IDLE (ambient stopped)")
    
    def on_tts_start(self):
        """Called when TTS starts speaking (auto-stops ambient)."""
        if self.state == AudioState.AMBIENT:
            # Cancel ambient synchronously, TTS takes over
            if self._ambient_task and not self._ambient_task.done():
                self._ambient_task.cancel()
                self._ambient_task = None
        self.state = AudioState.TTS
        logger.debug("🎤 AudioManager: State -> TTS")
    
    def on_tts_end(self):
        """Called when TTS finishes speaking."""
        if self.state == AudioState.TTS:
            self.state = AudioState.IDLE
            logger.debug("🎤 AudioManager: State -> IDLE (TTS done)")
    
    async def speak_injected(self, text: str) -> bool:
        """Speak text via /speak endpoint (external injection)."""
        if not self._tts_service or not self._task:
            logger.warning("AudioManager: No TTS service or task for injection")
            return False
        
        # Stop ambient if playing
        await self._cancel_ambient()
        await self._send_interruption()
        
        self.state = AudioState.INJECTED
        logger.info(f"💉 AudioManager: Injecting speech: {text[:50]}...")
        
        try:
            from pipecat.frames.frames import TextFrame, EndFrame
            # Queue text for TTS
            await self._task.queue_frame(TextFrame(text))
            self.state = AudioState.IDLE
            return True
        except Exception as e:
            logger.error(f"AudioManager: Injection failed: {e}")
            self.state = AudioState.IDLE
            return False
    
    async def play_sfx(self, sound_file: str):
        """Play a sound effect (short, can overlay other audio)."""
        # SFX doesn't change state - it overlays
        previous_state = self.state
        logger.info(f"🔊 AudioManager: Playing SFX: {sound_file}")
        await self._play_sound(sound_file)
        # State unchanged - SFX is fire-and-forget
    
    async def _play_sound(self, sound_file: str) -> float:
        """Play a sound file, return duration in seconds."""
        if not self._task:
            return 0
        
        from pipecat.frames.frames import OutputAudioRawFrame
        import wave
        
        file_path = os.path.join(os.path.dirname(__file__), sound_file)
        
        if not os.path.exists(file_path):
            logger.warning(f"AudioManager: Sound not found: {file_path}")
            return 0
        
        try:
            with wave.open(file_path, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                duration = n_frames / sample_rate
            
            with open(file_path, 'rb') as f:
                wav_data = f.read()
                audio_data = wav_data[44:]  # Skip WAV header
            
            frame = OutputAudioRawFrame(
                audio=audio_data,
                sample_rate=sample_rate,
                num_channels=1
            )
            await self._task.queue_frame(frame)
            logger.debug(f"🔊 AudioManager: Played {os.path.basename(sound_file)} ({duration:.1f}s)")
            
            return duration
            
        except Exception as e:
            logger.warning(f"AudioManager: Could not play {sound_file}: {e}")
            return 0


# Global audio manager instance
audio_manager = AudioManager()

# Store for captured video frames
_last_video_frame: Optional[bytes] = None


class VideoFrameCaptureObserver(BaseObserver):
    """Observer to capture video frames for later analysis."""
    
    _frame_count = 0
    
    async def on_push_frame(self, data):
        """Capture video frames when they pass through the pipeline."""
        global _last_video_frame
        
        frame = data.frame
        # Check for both InputImageRawFrame and UserImageRawFrame
        if isinstance(frame, (InputImageRawFrame, UserImageRawFrame)):
            try:
                from PIL import Image
                
                img = Image.frombytes('RGB', frame.size, frame.image)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                _last_video_frame = buffer.getvalue()
                
                # Log only every 60 frames (~1 per second at 60fps)
                VideoFrameCaptureObserver._frame_count += 1
                if VideoFrameCaptureObserver._frame_count % 60 == 0:
                    logger.debug(f"📸 Video frame captured ({len(_last_video_frame)} bytes)")
            except Exception as e:
                logger.warning(f"Video frame capture error: {e}")
_current_display_image: Optional[str] = None  # Path to currently displayed image
_image_display_until: float = 0  # Timestamp when to return to avatar
_has_greeted: bool = False  # Prevent duplicate greetings


async def _display_captured_frame(frame_bytes: bytes):
    """Display the captured video frame as output (show user what we're analyzing)."""
    global _current_task
    
    if not _current_task:
        return
    
    try:
        from pipecat.frames.frames import OutputImageRawFrame
        from PIL import Image
        import io
        
        # Load the JPEG frame and convert to RGB
        with Image.open(io.BytesIO(frame_bytes)) as img:
            img_rgb = img.convert('RGB')
            width, height = img_rgb.size
            raw_bytes = img_rgb.tobytes()
        
        frame = OutputImageRawFrame(
            image=raw_bytes,
            size=(width, height),
            format="RGB"
        )
        await _current_task.queue_frame(frame)
        logger.info(f"📸 Displayed captured frame: {width}x{height}")
    except Exception as e:
        logger.error(f"Failed to display captured frame: {e}")


async def send_default_avatar():
    """Send the default Vinston avatar image to video output (legacy, use display_manager)."""
    global _current_task
    
    if not _current_task or not os.path.exists(DEFAULT_AVATAR_PATH):
        return
    
    try:
        from pipecat.frames.frames import OutputImageRawFrame
        from PIL import Image
        
        with Image.open(DEFAULT_AVATAR_PATH) as img:
            img_rgb = img.convert('RGB')
            width, height = img_rgb.size
            raw_bytes = img_rgb.tobytes()
        
        frame = OutputImageRawFrame(
            image=raw_bytes,
            size=(width, height),
            format="RGB"
        )
        await _current_task.queue_frame(frame)
        logger.info(f"🐺 Default avatar sent: {width}x{height}")
    except Exception as e:
        logger.error(f"Failed to send default avatar: {e}")


# Global reference to avatar return task for cancellation
_avatar_return_task: Optional[asyncio.Task] = None


async def schedule_avatar_return(delay_seconds: int = 60):
    """Schedule returning to the default avatar after a delay."""
    try:
        await asyncio.sleep(delay_seconds)
        global _image_display_until
        _image_display_until = 0  # Allow avatar to be sent
        await send_default_avatar()
    except asyncio.CancelledError:
        logger.info("🖼️ Avatar return cancelled (new image displayed)")


async def _analyze_video_frame_impl(params) -> dict:
    global _avatar_return_task
    """
    Internal implementation - capture and analyze the current video frame.
    
    Mode is controlled by vision.mode in config.toml:
    - handoff: send to Clawdbot for analysis (default)
    - direct: feed image directly to voice LLM (Claude vision)
    - local: use a local vision model
    """
    global _last_video_frame
    
    if not _last_video_frame:
        return "No video frame available. Is the camera on?"
    
    vision_mode = get_vision_mode()
    logger.info(f"📸 Analyzing video frame (mode: {vision_mode})")
    
    try:
        import time
        import base64
        
        # Save frame to temp file
        timestamp = int(time.time())
        frame_path = f"/tmp/voxio-frame-{timestamp}.jpg"
        
        with open(frame_path, 'wb') as f:
            f.write(_last_video_frame)
        
        logger.info(f"📸 Saved video frame to {frame_path} ({len(_last_video_frame)} bytes)")
        
        if vision_mode == "handoff":
            # Hand off to Clawdbot for analysis
            class HandoffParams:
                def __init__(self):
                    self.arguments = {"task": f"Analyze the image at {frame_path} and describe what you see to the user"}
            
            handoff_result = await handoff_to_clawdbot(HandoffParams())
            
            return {
                "status": "handed_off",
                "frame_path": frame_path,
                "handoff_result": handoff_result
            }
            
        elif vision_mode == "direct":
            # Feed image directly to voice LLM via VisionImageRawFrame
            # The image will be added to the LLM context for inline analysis
            frame_b64 = base64.b64encode(_last_video_frame).decode('utf-8')
            
            return {
                "status": "analyzing",
                "mode": "direct",
                "image_data": frame_b64,
                "image_type": "image/jpeg",
                "instruction": "I've captured the image. Describe what you see in detail."
            }
            
        elif vision_mode == "local":
            # Use local vision model
            local_model = get_vision_local_model()
            
            # 1. Display the captured frame via DisplayManager
            await display_manager.show_capture(_last_video_frame)
            
            # 2. Start ambient sounds while processing
            await audio_manager.start_ambient()
            
            try:
                # 3. Run vision analysis
                description = await _analyze_with_local_model(frame_path, local_model)
            finally:
                # 4. Stop ambient sounds when done (success or error)
                await audio_manager.stop_ambient()
            
            # Return description as string - Pipecat passes this directly to LLM
            logger.info(f"📸 Local vision result: {description[:200] if description else 'None'}...")
            return f"[Vision Analysis]: {description}"
        else:
            return f"Error: Unknown vision mode: {vision_mode}"
            
    except Exception as e:
        logger.error(f"Video frame analysis error: {e}")
        return f"Error analyzing video frame: {str(e)}"


async def analyze_video_frame(params):
    """
    Capture and analyze the current video frame.
    Uses params.result_callback to return result to Pipecat LLM.
    """
    result = await _analyze_video_frame_impl(params)
    logger.info(f"📸 analyze_video_frame result: {str(result)[:200]}...")
    # Must use callback to pass result back to LLM!
    await params.result_callback(result)


async def _analyze_with_local_model(frame_path: str, model_name: str) -> str:
    """Analyze image with a local vision model."""
    logger.info(f"🔍 Analyzing with local model: {model_name}")
    
    try:
        if model_name == "moondream":
            # Use Moondream via transformers (lazy-loaded)
            model = _get_moondream_model()
            if model is None:
                return "Moondream model not available. Install with: uv add pipecat-ai[moondream]"
            
            from PIL import Image
            
            def run_moondream():
                image = Image.open(frame_path)
                image_embeds = model.encode_image(image)
                result = model.query(image_embeds, "Describe this image in detail.")
                return result.get("answer", str(result))
            
            # Run in thread to avoid blocking
            description = await asyncio.to_thread(run_moondream)
            logger.info(f"🌙 Moondream result: {description[:200] if description else 'None'}...")
            return description
                
        elif model_name == "llava":
            # LLaVA via ollama
            result = subprocess.run(
                ["ollama", "run", "llava", f"Describe this image: {frame_path}"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"LLaVA error: {result.stderr}"
                
        elif model_name == "florence-2":
            # Florence-2 - would need custom script
            return "Florence-2 not yet implemented. Use moondream or llava."
            
        else:
            return f"Unknown local model: {model_name}. Supported: moondream, llava, florence-2"
            
    except subprocess.TimeoutExpired:
        return "Local vision model timed out"
    except FileNotFoundError:
        return f"Local model '{model_name}' not installed or not in PATH"
    except Exception as e:
        logger.error(f"Local vision error: {e}")
        return f"Local vision error: {e}"


async def show_generated_image(params) -> dict:
    """
    Generate and display an image relevant to the conversation.
    Uses nano-banana (Gemini) to create the image.
    """
    global _current_display_image
    
    # Cancel current display and show avatar while generating
    await display_manager.show_avatar()
    
    prompt = params.arguments.get("prompt", "")
    if not prompt:
        return {"status": "error", "message": "No image prompt provided"}
    
    try:
        import time
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"/tmp/vinston-image-{timestamp}.png"
        
        logger.info(f"🎨 Generating image: {prompt[:50]}...")
        
        # Start ambient sounds while generating
        await audio_manager.start_ambient()
        
        try:
            # Call nano-banana script using async subprocess (non-blocking)
            proc = await asyncio.create_subprocess_exec(
                "uv", "run", NANO_BANANA_SCRIPT,
                "--prompt", prompt,
                "--filename", filename,
                "--resolution", "1K",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")}
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            # Create a result-like object for compatibility
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout.decode() if stdout else ""
                    self.stderr = stderr.decode() if stderr else ""
            result = Result(proc.returncode, stdout, stderr)
        finally:
            # Stop ambient sounds when done
            await audio_manager.stop_ambient()
        
        if result.returncode == 0 and os.path.exists(filename):
            _current_display_image = filename
            logger.info(f"✅ Image generated: {filename}")
            
            # Display via DisplayManager
            await display_manager.show_image(filename)
            return {"status": "success", "message": f"Image displayed: {prompt[:30]}..."}
        else:
            logger.error(f"Image generation failed: {result.stderr}")
            return {"status": "error", "message": "Failed to generate image"}
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Image generation timed out"}
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# GIF SEARCH AND DISPLAY
# ============================================================================

_gif_animation_task: Optional[asyncio.Task] = None


async def display_gif(params) -> dict:
    """
    Search for and display an animated GIF.
    Uses gifgrep to search Tenor/Giphy, downloads the GIF, and animates it.
    """
    query = params.arguments.get("query", "")
    if not query:
        return {"status": "error", "message": "No search query provided"}
    
    # Cancel current display and show avatar while searching
    await display_manager.show_avatar()
    
    try:
        import tempfile
        from PIL import Image
        
        logger.info(f"🎬 Searching for GIF: {query}")
        
        # Use gifgrep to search for GIF URL, then download
        with tempfile.TemporaryDirectory() as tmpdir:
            # Search for GIF URL
            result = subprocess.run(
                ["gifgrep", "search", query, "--max", "1"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            gif_url = result.stdout.strip()
            if not gif_url or not gif_url.startswith("http"):
                return {"status": "error", "message": f"No GIF found for '{query}'"}
            
            logger.info(f"🎬 Found GIF URL: {gif_url}")
            
            # Download the GIF
            gif_path = f"{tmpdir}/result.gif"
            async with aiohttp.ClientSession() as session:
                async with session.get(gif_url) as resp:
                    if resp.status != 200:
                        return {"status": "error", "message": f"Failed to download GIF: HTTP {resp.status}"}
                    gif_data = await resp.read()
                    with open(gif_path, 'wb') as f:
                        f.write(gif_data)
            
            if not os.path.exists(gif_path):
                return {"status": "error", "message": f"No GIF found for '{query}'"}
            
            logger.info(f"🎬 Downloaded GIF to: {gif_path}")
            
            # Load and decode GIF frames
            gif = Image.open(gif_path)
            frames = []
            durations = []
            
            try:
                while True:
                    # Convert frame to RGB
                    frame = gif.convert('RGB')
                    frames.append(frame.copy())
                    
                    # Get frame duration (default 100ms)
                    duration = gif.info.get('duration', 100)
                    durations.append(duration / 1000.0)  # Convert to seconds
                    
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
            
            logger.info(f"🎬 Loaded {len(frames)} frames")
            
            if not frames:
                return {"status": "error", "message": "Could not decode GIF frames"}
            
            # Display via DisplayManager
            await display_manager.show_gif(frames, durations)
            
            return {
                "status": "success",
                "message": f"Now playing GIF for '{query}' ({len(frames)} frames)",
                "frames": len(frames)
            }
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "GIF search timed out"}
    except Exception as e:
        logger.error(f"GIF display error: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# VOICE CHANGE TOOL
# ============================================================================

# Global reference to the current TTS service for voice changes
_current_tts_service = None


# Popular ElevenLabs voice options
VOICE_OPTIONS = {
    "roger": {"id": "CwhRBWXzGAHq8TQ4Fs17", "desc": "Roger - Professional male"},
    "jerry": {"id": "QzTKubutNn9TjrB7Xb2Q", "desc": "Jerry B - Brash, mischievous"},
    "adam": {"id": "pNInz6obpgDQGcFmaJgB", "desc": "Adam - Deep male"},
    "rachel": {"id": "21m00Tcm4TlvDq8ikWAM", "desc": "Rachel - Calm female"},
    "bella": {"id": "EXAVITQu4vr4xnSDxMaL", "desc": "Bella - Soft female"},
    "josh": {"id": "TxGEqnHWrfWFTfGW9XjX", "desc": "Josh - Young male"},
    "arnold": {"id": "VR6AewLTigWG4xSOukaG", "desc": "Arnold - Crisp male"},
    "sam": {"id": "yoZ06aMxZJJ28mfd3POQ", "desc": "Sam - Raspy male"},
}


async def change_voice(params) -> dict:
    """Change the TTS voice for the current session."""
    global _current_tts_service
    
    voice_name = params.arguments.get("voice", "").lower()
    
    if voice_name not in VOICE_OPTIONS:
        available = ", ".join(VOICE_OPTIONS.keys())
        return {"status": "error", "message": f"Unknown voice. Available: {available}"}
    
    voice_id = VOICE_OPTIONS[voice_name]["id"]
    voice_desc = VOICE_OPTIONS[voice_name]["desc"]
    
    try:
        # Directly update the voice_id on the ElevenLabs TTS service
        if _current_tts_service and hasattr(_current_tts_service, '_voice_id'):
            _current_tts_service._voice_id = voice_id
            logger.info(f"🎤 Voice changed to {voice_name} ({voice_id})")
            return {"status": "success", "message": f"Voice changed to {voice_desc}"}
        elif _current_tts_service:
            # Try setting via settings if available
            if hasattr(_current_tts_service, '_settings'):
                _current_tts_service._settings.voice_id = voice_id
                logger.info(f"🎤 Voice changed to {voice_name} ({voice_id}) via settings")
                return {"status": "success", "message": f"Voice changed to {voice_desc}"}
            else:
                return {"status": "error", "message": "Cannot update voice on current TTS service"}
        else:
            return {"status": "error", "message": "No active TTS service"}
            
    except Exception as e:
        logger.error(f"Failed to change voice: {e}")
        return {"status": "error", "message": str(e)}


# Tool definition for Pipecat/Claude
HANDOFF_TOOL = {
    "name": "handoff_to_clawdbot",
    "description": """Hand off a task to Clawdbot, the main AI assistant system.
Use this when the user asks for something that requires:
- Web searches or looking up current information
- Calendar access (checking schedule, creating events)
- Email operations (sending, reading)
- File system operations
- Smart home control
- Any task requiring external tools or APIs

Do NOT use for simple conversation, general knowledge questions, or tasks you can answer directly.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of what the user wants done. Include all relevant details from their request.",
            }
        },
        "required": ["task"],
    },
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

# ============================================================================
# DYNAMIC SYSTEM PROMPT - loads from workspace files
# ============================================================================

CLAWD_WORKSPACE = "/Users/visionik/clawd"

def load_workspace_file(filename: str) -> str:
    """Load a file from the Clawdbot workspace."""
    try:
        filepath = os.path.join(CLAWD_WORKSPACE, filename)
        with open(filepath, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
        return ""

def load_recent_memories() -> str:
    """Load recent memory files."""
    from datetime import datetime, timedelta
    memories = []
    memory_dir = os.path.join(CLAWD_WORKSPACE, "memory")
    
    try:
        # Get today and yesterday's dates
        today = datetime.now()
        for days_ago in range(3):  # Last 3 days
            date = today - timedelta(days=days_ago)
            filename = date.strftime("%Y-%m-%d.md")
            filepath = os.path.join(memory_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        memories.append(f"### Memory from {filename}:\n{content}")
    except Exception as e:
        logger.warning(f"Could not load memories: {e}")
    
    return "\n\n".join(memories) if memories else ""

def load_prompt_template() -> str:
    """Load the prompt template from VINSTON_PROMPT.md."""
    prompt_path = os.path.join(os.path.dirname(__file__), "VINSTON_PROMPT.md")
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Could not load VINSTON_PROMPT.md: {e}")
        return "You are Vinston Wolf, a voice AI assistant. Be helpful and concise."

def build_system_prompt() -> str:
    """Build the system prompt from template and workspace files."""
    identity = load_workspace_file("IDENTITY.md") or "- Name: Vinston Wolf\n- Creature: Wolf\n- Emoji: 🐺"
    soul = load_workspace_file("SOUL.md") or "- Keep replies concise and direct\n- Ask clarifying questions when needed"
    user = load_workspace_file("USER.md") or "- The user is the owner of this system"
    memories = load_recent_memories() or "(No recent memories loaded)"
    
    # Load template and replace placeholders
    template = load_prompt_template()
    prompt = template.replace("{{IDENTITY}}", identity)
    prompt = prompt.replace("{{SOUL}}", soul)
    prompt = prompt.replace("{{USER}}", user)
    prompt = prompt.replace("{{MEMORIES}}", memories)
    
    return prompt

# Build prompt at startup
logger.info("📜 Loading workspace files for system prompt...")
VINSTON_SYSTEM_PROMPT = build_system_prompt()
logger.info("✅ System prompt built with identity, soul, user profile, and memories")


# ============================================================================
# SIGHUP HANDLER - Reload config without restart
# ============================================================================

def reload_config(signum=None, frame=None):
    """Reload configuration on SIGHUP signal."""
    global VINSTON_SYSTEM_PROMPT, STT_PROVIDER, STT_MODEL, TTS_PROVIDER, TTS_VOICE, TTS_MODEL_PATH
    global VIDEO_IN_ENABLED, VIDEO_OUT_ENABLED, DEFAULT_AVATAR_PATH
    
    logger.info("🔄 SIGHUP received - reloading configuration...")
    
    # Reload .env file (for secrets)
    load_dotenv(override=True)
    
    # Reload TOML config
    load_config()
    
    # Update legacy globals from config
    STT_PROVIDER = get_stt_provider()
    STT_MODEL = get_stt_model()
    TTS_PROVIDER = get_tts_provider()
    TTS_VOICE = get_tts_voice()
    TTS_MODEL_PATH = get_tts_model_path()
    VIDEO_IN_ENABLED = get_video_in_enabled()
    VIDEO_OUT_ENABLED = get_video_out_enabled()
    DEFAULT_AVATAR_PATH = get_default_avatar_path()
    
    # Reload system prompt
    VINSTON_SYSTEM_PROMPT = build_system_prompt()
    
    logger.info("✅ Configuration reloaded!")
    logger.info(f"   STT: {STT_PROVIDER} ({STT_MODEL})")
    logger.info(f"   TTS: {TTS_PROVIDER} ({TTS_VOICE})")
    logger.info(f"   Video: in={VIDEO_IN_ENABLED}, out={VIDEO_OUT_ENABLED}")
    logger.info(f"   Handoff: {get_handoff_type()}")
    
    # Hot-swap TTS for active sessions
    try:
        from sessions import update_tts_for_all_sessions
        new_tts = create_tts_service()
        update_tts_for_all_sessions(new_tts)
    except Exception as e:
        logger.warning(f"Could not hot-swap TTS: {e}")

# Register SIGHUP handler
signal.signal(signal.SIGHUP, reload_config)
logger.info("📡 SIGHUP handler registered - send SIGHUP to reload config")


# ============================================================================
# MAIN BOT PIPELINE
# ============================================================================

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the Vinston voice assistant pipeline."""
    logger.info("Starting Vinston Wolf...")
    
    # Generate a unique session ID for this connection
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"🆔 New session: {session_id}")

    # Speech-to-Text: Configurable (cloud or local)
    stt = create_stt_service()

    # Text-to-Speech: Configurable (cloud or local)
    # Store reference globally for voice changes
    global _current_tts_service
    tts = create_tts_service()
    _current_tts_service = tts

    # LLM: Local (Anthropic + tools) or Gateway (full Clawdbot access)
    llm_mode = get_llm_mode()
    logger.info(f"🧠 LLM mode: {llm_mode}")
    
    if llm_mode == "gateway":
        # Gateway mode: All requests go through Clawdbot (full tool access)
        session_key = get_session_key() or "agent:main:voice"
        llm = GatewayLLMService(session_key=session_key)
        # No local tools registered - Gateway handles everything
        logger.info(f"🌐 Using Gateway LLM (session: {session_key})")
    else:
        # Local mode: VB's own Claude with limited tools + handoff
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
        )
        
        # Register tools (only in local mode)
        llm.register_function("handoff_to_clawdbot", handoff_to_clawdbot)
        llm.register_function("play_sound_effect", play_sound_effect)
        llm.register_function("analyze_video_frame", analyze_video_frame)
        llm.register_function("show_generated_image", show_generated_image)
        llm.register_function("display_gif", display_gif)
        # DISABLED: change_voice causes pipeline crashes due to dangling async tasks
        # llm.register_function("change_voice", change_voice)
        logger.info("🏠 Using Local LLM (Anthropic + tools)")

    # Conversation context with Vinston personality
    messages = [
        {
            "role": "system",
            "content": VINSTON_SYSTEM_PROMPT,
        },
    ]

    # Tools are passed via register_function + tools parameter with FunctionSchema
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    
    handoff_schema = FunctionSchema(
        name="handoff_to_clawdbot",
        description="Hand off a task to the main Clawdbot system when you need tools, file access, web search, calendar, email, or any action requiring external capabilities.",
        properties={
            "task": {
                "type": "string",
                "description": "A clear description of the task to perform"
            }
        },
        required=["task"]
    )
    
    sound_effect_schema = FunctionSchema(
        name="play_sound_effect",
        description="Generate and play a sound effect. Use for celebratory moments, emphasis, or when the user explicitly asks for a sound. Keep descriptions short and specific.",
        properties={
            "prompt": {
                "type": "string",
                "description": "Brief description of the sound effect, e.g. 'triumphant fanfare', 'sad trombone', 'drum roll', 'magical sparkle'"
            },
            "duration": {
                "type": "number",
                "description": "Duration in seconds (0.5 to 5.0, default 2.0)"
            }
        },
        required=["prompt"]
    )
    
    analyze_video_schema = FunctionSchema(
        name="analyze_video_frame",
        description="Look at what the user is showing via webcam or screen share. Use when they say 'look at this', 'what do you see', 'can you see this', etc.",
        properties={},
        required=[]
    )
    
    show_image_schema = FunctionSchema(
        name="show_generated_image",
        description="Generate and display an image relevant to the conversation. Use sparingly for visual emphasis, explaining concepts, or when discussing something visual. Image stays visible for about a minute.",
        properties={
            "prompt": {
                "type": "string",
                "description": "Description of the image to generate, e.g. 'a wolf howling at the moon', 'a diagram of a neural network', 'a cozy cabin in the woods'"
            }
        },
        required=["prompt"]
    )
    
    display_gif_schema = FunctionSchema(
        name="display_gif",
        description="Search for and display an animated GIF. Use when the user asks for a GIF, animation, or when an animated response would be fun/appropriate.",
        properties={
            "query": {
                "type": "string",
                "description": "Search query for the GIF, e.g. 'dancing cat', 'thumbs up', 'mind blown', 'wolf howling'"
            }
        },
        required=["query"]
    )
    
    change_voice_schema = FunctionSchema(
        name="change_voice",
        description="Change your speaking voice. Available voices: roger (professional), jerry (brash/mischievous), adam (deep), rachel (calm female), bella (soft female), josh (young male), arnold (crisp), sam (raspy).",
        properties={
            "voice": {
                "type": "string",
                "description": "Voice name: roger, jerry, adam, rachel, bella, josh, arnold, or sam"
            }
        },
        required=["voice"]
    )
    
    # Create context - with tools only in local mode
    if llm_mode == "gateway":
        # Gateway mode: no local tools, Gateway has all tools
        context = LLMContext(messages)
    else:
        # Local mode: register tool schemas
        context = LLMContext(messages, tools=ToolsSchema(standard_tools=[
            handoff_schema, 
            sound_effect_schema,
            analyze_video_schema,
            show_image_schema,
            display_gif_schema,
            # change_voice_schema  # DISABLED: causes pipeline crashes
        ]))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # RTVI processor for client communication
    rtvi = RTVIProcessor()

    # Build the pipeline: audio in → STT → LLM → TTS → audio out
    pipeline = Pipeline(
        [
            transport.input(),  # Audio/video from browser
            rtvi,  # RTVI protocol handling
            stt,  # Speech-to-text (OpenAI Whisper)
            user_aggregator,  # Collect user messages
            llm,  # Language model (Claude)
            tts,  # Text-to-speech (ElevenLabs)
            transport.output(),  # Audio to browser
            assistant_aggregator,  # Collect assistant responses
        ]
    )

    # Video frame capture observer
    video_observer = VideoFrameCaptureObserver()
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi), video_observer],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        global _current_task, _has_greeted
        logger.info(f"🐺 Client connected to Vinston (session: {session_id})")
        # Register this session for external access
        register_session(session_id, task, tts, context)
        # Set task reference for sound effects and video
        _current_task = task
        display_manager.set_task(task)
        audio_manager.set_task(task)
        audio_manager.set_tts(tts)
        # Send default avatar ONLY if not already displaying something
        # (preserves display state across reconnects)
        if display_manager.state == DisplayState.AVATAR:
            try:
                await display_manager.show_avatar()
            except Exception as e:
                logger.error(f"Failed to send avatar on connect: {e}")
        else:
            logger.info(f"🖼️ Preserving display state: {display_manager.state.name}")
        
        # Only greet once per session
        if _has_greeted:
            logger.info("Skipping duplicate greeting")
            return
        _has_greeted = True
        
        # Greet the user when they connect (pick random greeting)
        greeting = get_random_greeting()
        logger.info(f"🐺 Using greeting: {greeting}")
        messages.append(
            {
                "role": "system",
                "content": f"A user just connected. Say exactly: '{greeting}'",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        global _current_task, _has_greeted
        logger.info(f"Client disconnected from Vinston (session: {session_id})")
        unregister_session(session_id)
        _current_task = None
        _has_greeted = False  # Reset for next connection
        await audio_manager.interrupt_all()  # Stop any playing audio
        await task.cancel()

    @transport.event_handler("on_user_started_speaking")
    async def on_user_started_speaking(transport):
        """Stop all audio when user starts speaking."""
        await audio_manager.interrupt_all()
        logger.info("🔇 User speaking - audio interrupted")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def get_cloudflare_ice_servers():
    """Fetch ICE servers from Cloudflare TURN API."""
    import aiohttp
    
    cf_token = os.getenv("CF_TURN_TOKEN")
    cf_key_id = os.getenv("CF_TURN_KEY_ID")
    
    if not cf_token or not cf_key_id:
        logger.warning("Cloudflare TURN not configured, using STUN only")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{cf_key_id}/credentials/generate-ice-servers"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={
                    "Authorization": f"Bearer {cf_token}",
                    "Content-Type": "application/json"
                },
                json={"ttl": 86400}
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    logger.info("✅ Got Cloudflare TURN credentials")
                    return data.get("iceServers", [])
                else:
                    logger.error(f"Failed to get TURN credentials: {resp.status}")
                    return [{"urls": ["stun:stun.l.google.com:19302"]}]
    except Exception as e:
        logger.error(f"Error fetching TURN credentials: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    
    # Get Cloudflare TURN credentials for WebRTC
    ice_servers = await get_cloudflare_ice_servers()
    logger.info(f"🧊 ICE servers configured: {len(ice_servers)} servers")

    # Configure transport for WebRTC with VAD
    # Set audio output sample rate based on TTS provider
    audio_sample_rate = 22050 if TTS_PROVIDER == "piper" else 24000
    logger.info(f"🔊 Audio output sample rate: {audio_sample_rate}Hz (TTS: {TTS_PROVIDER})")
    
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=audio_sample_rate,
            video_in_enabled=VIDEO_IN_ENABLED,
            video_out_enabled=VIDEO_OUT_ENABLED,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,  # Wait 300ms of silence before considering turn complete
                )
            ),
            # Cloudflare TURN for NAT traversal
            ice_servers=ice_servers,
        ),
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,  # TODO: Daily video output crashes - investigate
            video_out_enabled=False,
            camera_in_enabled=VIDEO_IN_ENABLED,
            video_in_enabled=VIDEO_IN_ENABLED,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                )
            ),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def get_transport_type():
    """Get transport type from config."""
    return get_config("transport.type", "webrtc")

def get_transport_port():
    """Get transport port from config."""
    return get_config("transport.port", 8086)

def get_transport_host():
    """Get transport host from config."""
    return get_config("transport.host", None)

def get_transport_proxy():
    """Get transport proxy from config."""
    return get_config("transport.proxy", None)

def get_transport_esp32():
    """Get ESP32 compatibility mode from config."""
    return get_config("transport.esp32", False)

def get_transport_room():
    """Get Daily room URL from config (direct connection)."""
    return get_config("transport.room", None)

def get_transport_folder():
    """Get downloads folder from config."""
    return get_config("transport.folder", None)

def get_transport_verbose():
    """Get verbose logging from config."""
    return get_config("transport.verbose", False)


if __name__ == "__main__":
    import sys
    
    # Helper to check if arg exists
    def has_arg(*args):
        return any(arg in sys.argv for arg in args)
    
    # Strip custom args that Pipecat doesn't know about
    # (they're already parsed by _get_cli_arg and _has_cli_flag at module load)
    custom_args = [
        '--config', '-c',
        '--llm-mode',
        '--session', '-s',
        '--session-label',
        '--require-existing',
        '--reset-session',
    ]
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in custom_args:
            # Skip this arg and its value (if it takes one)
            if arg not in ['--require-existing', '--reset-session']:  # flags without values
                i += 1  # skip the value too
        else:
            filtered_argv.append(arg)
        i += 1
    sys.argv = filtered_argv
    
    # Add config values as defaults if not specified on CLI
    if not has_arg('-t', '--transport'):
        transport_type = get_transport_type()
        sys.argv.extend(['-t', transport_type])
        logger.info(f"📡 Using transport from config: {transport_type}")
    
    if not has_arg('-p', '--port'):
        port = get_transport_port()
        sys.argv.extend(['--port', str(port)])
        logger.info(f"📡 Using port from config: {port}")
    
    if not has_arg('--host'):
        host = get_transport_host()
        if host:
            sys.argv.extend(['--host', host])
            logger.info(f"📡 Using host from config: {host}")
    
    if not has_arg('-x', '--proxy'):
        proxy = get_transport_proxy()
        if proxy:
            sys.argv.extend(['--proxy', proxy])
            logger.info(f"📡 Using proxy from config: {proxy}")
    
    if not has_arg('--esp32'):
        if get_transport_esp32():
            sys.argv.append('--esp32')
            logger.info("📡 ESP32 compatibility enabled from config")
    
    if not has_arg('-d', '--direct'):
        room = get_transport_room()
        if room:
            sys.argv.extend(['--direct'])
            # Set DAILY_ROOM_URL env var for the room
            os.environ['DAILY_ROOM_URL'] = room
            logger.info(f"📡 Using Daily room from config: {room}")
    
    if not has_arg('-f', '--folder'):
        folder = get_transport_folder()
        if folder:
            sys.argv.extend(['--folder', folder])
            logger.info(f"📡 Using folder from config: {folder}")
    
    if not has_arg('-v', '--verbose'):
        if get_transport_verbose():
            sys.argv.append('--verbose')
            logger.info("📡 Verbose logging enabled from config")
    
    from pipecat.runner.run import main
    main()
