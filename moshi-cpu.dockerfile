FROM python:3.11-slim

# Install dependencies
RUN apt-get update 
    # apt-get install -y ffmpeg

# Install Python deps
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install asyncio websockets numpy huggingface_hub sentencepiece

# Copy server
WORKDIR /app

RUN pip install uv

COPY moshi/moshi /app/moshi/

RUN cd /app/moshi/moshi && uv run sync

COPY . .

EXPOSE 9800 

CMD ["uv","run" "server.py"]
