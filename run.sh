
docker build -t moshi  -f moshi-cpu.dockerfile .
docker run -v "$(pwd):/root/.cache/huggingface/hub" --env  "HF_HUB_ENABLE_HF_TRANSFER=1" -p 9800:9800 moshi