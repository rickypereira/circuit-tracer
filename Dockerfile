# Use a pre-built PyTorch GPU image from Google's Deep Learning Containers.
# We're aiming for:
# - PyTorch 2.4+ (your requirements show 2.7.0)
# - Python 3.11
# - CUDA 12.4 (or compatible version provided by Google)
#
# As per Google's documentation and recent releases, a Pytorch 2.4 with Python 3.11 and CUDA 12.4 is available.
# The exact tag might change over time, so always check the official Deep Learning Containers documentation:
# https://cloud.google.com/deep-learning-containers/docs/choosing-container
# You might find something like: us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-4.py311:latest
# Or a newer version as PyTorch advances. Let's use a very recent one that aligns with your CUDA 12.4.
# The '2-7.py311' part is speculative as of my last update, but matches your torch version and Python.
# You might need to check the exact available tags on Google's Deep Learning Containers list.
# A safe bet is often 'pytorch-gpu.<latest_stable_pytorch_version>.py311:latest' or specifically matching.
# For CUDA 12.4, PyTorch 2.4.x is commonly paired with it. Let's aim for a PyTorch 2.4.
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310:latest

# Set working directory inside the container
WORKDIR /app
COPY . .
ENTRYPOINT ["python", "scripts/train_transcoder.py"]

# Optional: If you need specific environment variables set for your script, add them here.
# ENV MY_CUSTOM_VAR="some_value"