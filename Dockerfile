FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310:latest

# Set working directory inside the container
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install "numpy==1.26.4" \
    && pip install --no-cache-dir -r requirements.txt
ARG TRANSCODERS_HF_TOKEN
ENV TRANSCODERS_HF_TOKEN=$TRANSCODERS_HF_TOKEN
ENV PYTHONPATH=/app:${PYTHONPATH}
ENTRYPOINT ["python3", "scripts/train_transcoder.py"]