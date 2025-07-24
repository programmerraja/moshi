import asyncio
import websockets
import numpy as np

# import soundfile as sf
import uuid
import os
import logging
from typing import AsyncGenerator
import time
import torch

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Optimize model selection and device
def get_optimal_device_and_compute_type():
    """Determine the best device and compute type for the system"""
    if torch.cuda.is_available():
        return "cuda", "float16"
    else:
        return "cpu", "int8"


device, compute_type = get_optimal_device_and_compute_type()
logger.info(f"Using device: {device}, compute_type: {compute_type}")

# Use tiny model for fastest processing - can change to "base" if accuracy is more important
model = WhisperModel(
    "tiny",
    device=device,
    compute_type=compute_type,
    num_workers=1,  # Limit workers for better performance
)


async def run_stt(audio: bytes, language: str = "en") -> AsyncGenerator[str, None]:
    """Optimized Whisper STT with performance tweaks"""
    if not model:
        logger.error("Whisper model not available")
        yield "Whisper model not available"
        return

    # PCM int16 -> float32 in [-1, 1]
    audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    # # Skip very short audio to avoid processing overhead
    # if len(audio_float) < 1600:  # Less than 0.1 seconds at 16kHz
    #     return

    try:
        segments, _ = await asyncio.to_thread(
            model.transcribe,
            audio_float,
            language=language,
            beam_size=1,  # Faster than default beam_size=5
            temperature=0.0,  # Deterministic and faster
            condition_on_previous_text=False,  # Faster processing
            word_timestamps=False,  # Disable word-level timestamps for speed
            vad_filter=False,  # Enable VAD to skip silent parts
            # vad_parameters=dict(min_silence_duration_ms=1000)  # Aggressive VAD
        )

        text = ""
        for segment in segments:
            text += f"{segment.text} "
        
        logger.info(f"text {text}")
        
        if text.strip():
            yield text.strip()

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        yield str(e)


async def handler(websocket):
    logger.info("New WebSocket connection established")

    try:
        async for message in websocket:
            start_time = time.perf_counter()

            try:
                async for frame in run_stt(message):
                    print(f"Sending frame: {frame}")
                    await websocket.send(frame)

                processing_time = time.perf_counter() - start_time
                logger.info(f"Total processing time: {processing_time:.3f} seconds")

            except Exception as e:
                logger.error(f"Handler error: {e}")
                await websocket.send(f"ERROR: {str(e)}")

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")


async def main():
    # Pre-warm the model with dummy audio for faster first transcription
    logger.info("Pre-warming model...")
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    try:
        list(model.transcribe(dummy_audio, beam_size=1))
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

    async with websockets.serve(handler, "0.0.0.0", 9800):
        logger.info("Whisper WebSocket server listening on ws://0.0.0.0:9800")
        logger.info(f"Model: tiny, Device: {device}, Compute: {compute_type}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
