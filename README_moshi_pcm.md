# Simple Moshi PCM WebSocket Server

A simplified WebSocket server that accepts PCM audio input and returns PCM audio output using the Moshi model.

## Features

- Simple WebSocket-based PCM audio processing
- Real-time audio streaming with Moshi
- Minimal dependencies and clean code structure
- Error handling with silence fallback
- Performance logging

## Installation

1. Install dependencies:
```bash
pip install -r requirements_moshi_pcm.txt
```

2. Make sure you have the Moshi package in your project directory.

## Usage

### Starting the Server

```bash
python moshi_simple_pcm_server.py --host 0.0.0.0 --port 8999 --model kyutai/moshi-7b-202409 --device cuda
```

### Arguments

- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8999)
- `--model`: Model path or HuggingFace repo (default: kyutai/moshi-7b-202409)
- `--device`: Device to use (cuda/cpu, default: cuda)

### Testing

Run the test client to verify the server:

```bash
python test_moshi_pcm_client.py
```

## Protocol

The server expects:
- **Input**: Raw PCM audio data as bytes (float32 format)
- **Output**: Processed PCM audio data as bytes (float32 format)

### PCM Format
- Sample rate: 24kHz (Moshi's default)
- Format: 32-bit float (little-endian)
- Channels: Mono

## Example Usage

```python
import asyncio
import websockets
import struct
import numpy as np

async def send_audio():
    async with websockets.connect('ws://localhost:8999') as websocket:
        # Generate test audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        pcm_bytes = struct.pack(f'<{len(audio)}f', *audio)
        
        # Send audio
        await websocket.send(pcm_bytes)
        
        # Receive processed audio
        response = await websocket.recv()
        print(f"Received {len(response)} bytes")

asyncio.run(send_audio())
```

## Performance Notes

- The server processes audio in frames based on Moshi's frame rate
- First frame is used for initialization and may be skipped
- Processing time is logged for each request
- Errors result in silence being returned

## Troubleshooting

1. **Import errors**: Make sure the Moshi package is in your Python path
2. **CUDA errors**: Use `--device cpu` if CUDA is not available
3. **Model loading errors**: Check your internet connection for HF model downloads
4. **Memory issues**: Consider using a smaller model or CPU inference 