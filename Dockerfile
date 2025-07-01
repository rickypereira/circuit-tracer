FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310:latest

# Set working directory inside the container
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install "numpy==1.26.4" \
    && pip install --no-cache-dir -r requirements.txt
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
ENV PYTHONPATH=/app:${PYTHONPATH}
ENTRYPOINT ["torchrun", "--nproc_per_node", "gpu", "-m", "scripts.train_transcoder"]
CMD ["--model_name", "gemma-2-9b", \
     "--distribute_modules", \
     "--batch_size", "1", \
     "--layers_partition_size", "8", \
     "--grad_acc_steps", "128", \
     "--ctx_len", "2048", \
     "--k", "192", \
     "--load_in_4bit", \
     "--micro_acc_steps", "32", \
     "--log_to_wandb", \
     "--dataset_train_size", "4000"]