FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310:latest

# Set working directory inside the container
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "scripts/train_transcoder.py"]