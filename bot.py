#
# Voxio Bot (voxio.bot)
# Real-time voice + video AI using Pipecat + Claude + ElevenLabs
# With Clawdbot bidirectional communication support
#

"""
Voxio Bot - Voice & video AI assistant.

Run locally with:
    uv run python run_auth.py

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
# TOML CONFIGURATION
# ============================================================================

CONFIG_PATH = Path(__file__).parent / "config.toml"
_config: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    """Load configuration from config.toml."""
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
from pipecat.frames.frames import LLMRunFrame, TextFrame, TTSSpeakFrame
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

# Conditional imports for local Whisper
try:
    from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTServiceMLX, Model as WhisperModel, MLXModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Local Whisper not available. Install with: uv add pipecat-ai[whisper] or pipecat-ai[mlx-whisper]")
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# ============================================================================
# CONFIGURATION (from config.toml, with .env fallback for secrets)
# ============================================================================

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
            
            # Stop ambient sounds before speaking (interrupt them)
            if is_ambient_sounds_playing():
                stop_ambient_sounds()
                logger.info("🔇 /speak: Stopped ambient sounds")
            
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


def stop_ambient_sounds():
    """Stop the ambient sound loop if running."""
    global _ambient_sound_task, _ambient_sounds_playing
    
    if _ambient_sound_task and not _ambient_sound_task.done():
        _ambient_sound_task.cancel()
        logger.info("🔇 Ambient sounds stopped")
    
    _ambient_sound_task = None
    _ambient_sounds_playing = False


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
        start_ambient_sounds()
        
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
        # Run clawdbot wake command
        logger.info(f"🤖 Handing off to Clawdbot: {task[:100]}...")
        
        result = subprocess.run(
            ["clawdbot", "wake", "--mode", "now", "--text", full_message],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            logger.info("✅ Clawdbot wake successful")
            # Speak acknowledgment first, then play typing sound
            await speak_then_play_handoff_sound()
            # Return empty message since we already spoke
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

# Store for captured video frames
_last_video_frame: Optional[bytes] = None
_current_display_image: Optional[str] = None  # Path to currently displayed image
_image_display_until: float = 0  # Timestamp when to return to avatar
_has_greeted: bool = False  # Prevent duplicate greetings


async def send_default_avatar():
    """Send the default Vinston avatar image to video output."""
    global _current_task, _image_display_until
    
    # Don't send avatar if we're still displaying a generated image
    import time
    if time.time() < _image_display_until:
        return
    
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


async def schedule_avatar_return(delay_seconds: int = 60):
    """Schedule returning to the default avatar after a delay."""
    await asyncio.sleep(delay_seconds)
    global _image_display_until
    _image_display_until = 0  # Allow avatar to be sent
    await send_default_avatar()


async def analyze_video_frame(params) -> dict:
    """
    Analyze what the user is showing via webcam/screen share.
    Captures the current video frame and describes what's visible.
    """
    global _last_video_frame
    
    if not _last_video_frame:
        return {"status": "error", "message": "No video frame available. Is the camera on?"}
    
    try:
        import base64
        
        # Convert frame to base64 for Claude
        frame_b64 = base64.b64encode(_last_video_frame).decode('utf-8')
        
        # Return the frame data for Claude to analyze inline
        # The actual analysis happens in the LLM with vision capability
        return {
            "status": "success",
            "message": "Frame captured. Analyzing what I see...",
            "frame_data": frame_b64,
            "frame_type": "image/jpeg"
        }
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        return {"status": "error", "message": str(e)}


async def show_generated_image(params) -> dict:
    """
    Generate and display an image relevant to the conversation.
    Uses nano-banana (Gemini) to create the image.
    """
    global _current_display_image
    
    prompt = params.arguments.get("prompt", "")
    if not prompt:
        return {"status": "error", "message": "No image prompt provided"}
    
    try:
        import tempfile
        import time
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"/tmp/vinston-image-{timestamp}.png"
        
        logger.info(f"🎨 Generating image: {prompt[:50]}...")
        
        # Call nano-banana script
        result = subprocess.run(
            ["uv", "run", NANO_BANANA_SCRIPT, 
             "--prompt", prompt, 
             "--filename", filename,
             "--resolution", "1K"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")}
        )
        
        if result.returncode == 0 and os.path.exists(filename):
            _current_display_image = filename
            logger.info(f"✅ Image generated: {filename}")
            
            # Queue image to video output if we have a task
            # DISABLED: OutputImageRawFrame crashes WebRTC mid-session
            # TODO: Investigate proper video frame streaming or use URL-based display
            if False and _current_task:
                try:
                    from pipecat.frames.frames import OutputImageRawFrame
                    from PIL import Image
                    import time
                    
                    global _image_display_until
                    
                    # Small delay to let any pending operations complete
                    await asyncio.sleep(0.1)
                    
                    # Load and convert image to raw RGB bytes
                    with Image.open(filename) as img:
                        img_rgb = img.convert('RGB')
                        width, height = img_rgb.size
                        raw_bytes = img_rgb.tobytes()
                    
                    # Check task is still valid
                    if not _current_task:
                        logger.warning("Task became None before queuing image")
                        return {"status": "success", "message": f"Image generated but session ended: {prompt[:30]}..."}
                    
                    # Send image frame to video output
                    frame = OutputImageRawFrame(
                        image=raw_bytes,
                        size=(width, height),
                        format="RGB"
                    )
                    await _current_task.queue_frame(frame)
                    logger.info(f"📺 Image queued to video output: {width}x{height}")
                    
                    # Set timer to keep image up for 60 seconds
                    _image_display_until = time.time() + 60
                    
                    # Schedule return to avatar after 60 seconds (with error handling)
                    try:
                        asyncio.create_task(schedule_avatar_return(60))
                    except Exception as e:
                        logger.warning(f"Could not schedule avatar return: {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to queue image to video: {e}")
                    # Don't fail the whole function - image was still generated
            else:
                logger.info(f"📺 Image generated (no active task): {filename}")
                
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

    # LLM: Anthropic Claude with tools
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-20250514",
    )
    
    # Register tools
    llm.register_function("handoff_to_clawdbot", handoff_to_clawdbot)
    llm.register_function("play_sound_effect", play_sound_effect)
    llm.register_function("analyze_video_frame", analyze_video_frame)
    llm.register_function("show_generated_image", show_generated_image)
    # DISABLED: change_voice causes pipeline crashes due to dangling async tasks
    # llm.register_function("change_voice", change_voice)

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
    
    context = LLMContext(messages, tools=ToolsSchema(standard_tools=[
        handoff_schema, 
        sound_effect_schema,
        analyze_video_schema,
        show_image_schema,
        # change_voice_schema  # DISABLED: causes pipeline crashes
    ]))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # RTVI processor for client communication
    rtvi = RTVIProcessor()

    # Build the pipeline: audio in → STT → LLM → TTS → audio out
    pipeline = Pipeline(
        [
            transport.input(),  # Audio from browser
            rtvi,  # RTVI protocol handling
            stt,  # Speech-to-text (OpenAI Whisper)
            user_aggregator,  # Collect user messages
            llm,  # Language model (Claude)
            tts,  # Text-to-speech (ElevenLabs)
            transport.output(),  # Audio to browser
            assistant_aggregator,  # Collect assistant responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        global _current_task, _has_greeted
        logger.info(f"🐺 Client connected to Vinston (session: {session_id})")
        # Register this session for external access
        register_session(session_id, task, tts, context)
        # Set task reference for sound effects and video
        _current_task = task
        # Send default avatar to video output (wrapped in try/except)
        try:
            await send_default_avatar()
        except Exception as e:
            logger.error(f"Failed to send avatar on connect: {e}")
        
        # Only greet once per session
        if _has_greeted:
            logger.info("Skipping duplicate greeting")
            return
        _has_greeted = True
        
        # Greet the user when they connect
        messages.append(
            {
                "role": "system",
                "content": "A user just connected. Say exactly: 'Vinston here. What needs fixing?'",
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
        stop_ambient_sounds()  # Stop any playing ambient sounds
        await task.cancel()

    @transport.event_handler("on_user_started_speaking")
    async def on_user_started_speaking(transport):
        """Stop ambient sounds when user starts speaking."""
        if is_ambient_sounds_playing():
            stop_ambient_sounds()
            logger.info("🔇 User speaking - stopped ambient sounds")

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
            camera_out_enabled=False,
            video_out_enabled=False,
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


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
