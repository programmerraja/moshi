import asyncio
import websockets
import numpy as np
import struct
import time


async def test_moshi_pcm_server():
    """Test client for Moshi PCM WebSocket server"""
    uri = "ws://localhost:8999"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Generate test PCM data (1 second of sine wave at 440Hz)
            sample_rate = 24000  # Moshi uses 24kHz
            duration = 1.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Convert to bytes
            pcm_bytes = struct.pack(f'<{len(audio_data)}f', *audio_data)
            
            print(f"Sending {len(pcm_bytes)} bytes of PCM data...")
            start_time = time.time()
            
            # Send PCM data
            await websocket.send(pcm_bytes)
            
            # Receive response
            response = await websocket.recv()
            
            processing_time = time.time() - start_time
            print(f"Received {len(response)} bytes in {processing_time:.3f} seconds")
            
            # Convert response back to audio
            if len(response) % 4 == 0:
                response_audio = struct.unpack(f'<{len(response)//4}f', response)
                print(f"Response audio: {len(response_audio)} samples")
                print(f"Response audio range: {min(response_audio):.3f} to {max(response_audio):.3f}")
            else:
                print("Invalid response format")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_moshi_pcm_server()) 