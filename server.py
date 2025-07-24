import asyncio
import websockets
import numpy as np
import struct
import time
import logging
import torch
import argparse
import sys
import os

# Add the moshi package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'moshi/moshi'))

from moshi.moshi.models import loaders, MimiModel, LMModel, LMGen
from moshi.moshi.run_inference import get_condition_tensors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoshiPcmServer:
    def __init__(self, model_path=None, device="cuda"):
        self.device = device
        self.model = None
        self.mimi = None
        self.lm_gen = None
        self.frame_size = None
        self.initialized = False
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load Moshi model from path or HF repo"""
        try:
            logger.info("Loading Moshi model...")
            
            # Load checkpoint info
            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
                model_path, None, None, None
            )
            
            # Load Mimi model
            self.mimi = checkpoint_info.get_mimi(device=self.device)
            logger.info("Mimi model loaded")
            
            # Load LM model
            lm = checkpoint_info.get_moshi(device=self.device, dtype=torch.bfloat16)
            logger.info("LM model loaded")
            
            # Setup LM generator
            condition_tensors = get_condition_tensors(
                checkpoint_info.model_type, lm, batch_size=1, cfg_coef=1.0
            )
            self.lm_gen = LMGen(lm, cfg_coef=1.0, condition_tensors=condition_tensors)
            
            # Calculate frame size
            self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            
            # Setup streaming
            self.mimi.streaming_forever(1)
            self.lm_gen.streaming_forever(1)
            
            # Warmup
            self.warmup()
            
            self.initialized = True
            logger.info(f"Model loaded successfully. Frame size: {self.frame_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def warmup(self):
        """Warmup the model with dummy data"""
        logger.info("Warming up model...")
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is not None:
                    _ = self.mimi.decode(tokens[:, 1:])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("Model warmed up")
    
    async def process_pcm(self, pcm_data: bytes) -> bytes:
        """Process PCM input and return PCM output"""
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        
        try:
            # Convert bytes to float32 PCM samples
            if len(pcm_data) % 4 != 0:
                raise ValueError(f"Invalid PCM data length: {len(pcm_data)}")
            
            pcm_samples = struct.unpack(f'<{len(pcm_data)//4}f', pcm_data)
            
            # Ensure we have enough samples for a complete frame
            if len(pcm_samples) < self.frame_size:
                # Pad with zeros if needed
                pcm_samples = list(pcm_samples) + [0.0] * (self.frame_size - len(pcm_samples))
            
            # Take only the frame_size samples
            chunk = pcm_samples[:self.frame_size]
            
            # Convert to tensor
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)[None, None]
            
            # Encode with Mimi
            codes = self.mimi.encode(chunk_tensor)
            
            # Process through LM
            output_pcm = None
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is not None:
                    assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                    output_pcm = self.mimi.decode(tokens[:, 1:])
                    output_pcm = output_pcm.cpu()
            
            if output_pcm is None:
                # Return silence if no output
                output_pcm = torch.zeros(1, 1, self.frame_size, dtype=torch.float32)
            
            # Convert tensor to bytes
            pcm_output = output_pcm[0, 0].numpy()
            pcm_bytes = struct.pack(f'<{len(pcm_output)}f', *pcm_output)
            
            return pcm_bytes
            
        except Exception as e:
            logger.error(f"Error processing PCM: {e}")
            # Return silence on error
            silence = struct.pack(f'<{self.frame_size}f', *([0.0] * self.frame_size))
            return silence


async def handler(websocket, path, moshi_server):
    """WebSocket handler for PCM processing"""
    logger.info("New WebSocket connection established")
    
    try:
        async for message in websocket:
            start_time = time.perf_counter()
            
            try:
                # Process PCM input
                output_pcm = await moshi_server.process_pcm(message)
                
                # Send PCM output back
                await websocket.send(output_pcm)
                
                processing_time = time.perf_counter() - start_time
                logger.info(f"PCM processing time: {processing_time:.3f} seconds")
                
            except Exception as e:
                logger.error(f"Handler error: {e}")
                # Send silence on error
                silence = struct.pack(f'<{moshi_server.frame_size}f', *([0.0] * moshi_server.frame_size))
                await websocket.send(silence)
    
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Simple Moshi PCM WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8999, type=int, help="Port to bind to")
    parser.add_argument("--model", default="kyutai/moshi-7b-202409", help="Model path or HF repo")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize Moshi server
    moshi_server = MoshiPcmServer(model_path=args.model, device=args.device)
    
    # Create WebSocket server
    async with websockets.serve(
        lambda ws, path: handler(ws, path, moshi_server), 
        args.host, 
        args.port
    ):
        logger.info(f"Moshi PCM WebSocket server listening on ws://{args.host}:{args.port}")
        logger.info(f"Model: {args.model}, Device: {args.device}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main()) 