"""
Local Piper TTS Service for Pipecat.
Uses piper-tts directly without requiring an HTTP server.
"""

import asyncio
import io
import wave
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class PiperLocalTTSService(TTSService):
    """Local Piper TTS service using piper-tts directly.
    
    This runs Piper inference locally without requiring an HTTP server.
    """

    def __init__(
        self,
        *,
        model_path: str,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the local Piper TTS service.

        Args:
            model_path: Path to the .onnx voice model file
            config_path: Path to the .onnx.json config file (auto-detected if None)
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        # Piper outputs 22050Hz audio for high-quality models
        super().__init__(sample_rate=22050, **kwargs)

        self._model_path = model_path
        self._config_path = config_path or f"{model_path}.json"
        
        self._voice: Optional["PiperVoice"] = None
        
        self._load()

    def _load(self):
        """Load the Piper voice model."""
        try:
            from piper import PiperVoice

            logger.info(f"🔊 Loading Piper voice model: {self._model_path}")
            self._voice = PiperVoice.load(
                self._model_path,
                config_path=self._config_path,
                use_cuda=False,  # macOS doesn't have CUDA, use CPU/CoreML
            )
            
            # Get actual sample rate from the model
            if self._voice.config and hasattr(self._voice.config, 'sample_rate'):
                self._sample_rate = self._voice.config.sample_rate
                logger.info(f"🔊 Piper model sample rate: {self._sample_rate}Hz")
                
            logger.info("✅ Piper voice loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Piper voice: {e}")
            self._voice = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using local Piper.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if not self._voice:
            yield ErrorFrame(error="Piper voice model not loaded")
            return

        logger.debug(f"🔊 Piper TTS: [{text[:50]}...]")
        
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            
            yield TTSStartedFrame()

            # Stream audio chunks as they're generated (per sentence)
            import queue
            import threading
            
            audio_queue: queue.Queue = queue.Queue()
            synthesis_done = threading.Event()
            
            def synthesize_thread():
                try:
                    for chunk in self._voice.synthesize(text):
                        audio_queue.put(chunk.audio_int16_bytes)
                finally:
                    synthesis_done.set()
            
            # Start synthesis in background thread
            thread = threading.Thread(target=synthesize_thread, daemon=True)
            thread.start()
            
            first_chunk = True
            
            # Stream chunks as they become available
            while not synthesis_done.is_set() or not audio_queue.empty():
                try:
                    raw_audio = audio_queue.get(timeout=0.1)
                    
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False
                    
                    # Yield in smaller chunks for smoother streaming
                    chunk_size = 4096
                    for i in range(0, len(raw_audio), chunk_size):
                        chunk = raw_audio[i:i + chunk_size]
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                        )
                except queue.Empty:
                    continue
            
            thread.join(timeout=1.0)

        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            yield ErrorFrame(error=f"Piper TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
            logger.debug(f"🔊 Piper TTS complete: [{text[:30]}...]")
