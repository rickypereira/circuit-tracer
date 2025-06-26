import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import HfApi
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal

# --- Import from sparsify ---
from sparsify import TranscoderConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_PROJECT_PATH=Path(f"/home/rickpereira").resolve()

# --- Helper Functions ---
def setup_environment():
    print("Ensuring environment is set up...")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Hugging Face token not found.")
        sys.exit(1)

    if hf_token:
        print("Hugging Face token found in environment variables.")
        try:
            HfApi().whoami()
            print("Successfully authenticated with Hugging Face Hub.")
        except Exception as e:
            print(f"Error validating Hugging Face token: {e}")
            print(f"Please ensure your Hugging Face token (HF_TOKEN) is valid.")
            sys.exit(1)
    else:
        print("Hugging Face token (HF_TOKEN or HUGGING_FACE_HUB_TOKEN) not found in environment variables.")
        print("Please set HF_TOKEN environment variable when running the Docker container.")
        sys.exit(1)

def setup(rank, world_size):
    """Initializes the distributed process group."""
    # These environment variables are usually set by torchrun/torch.distributed.launch
    # but good to have as fallbacks or for local testing.
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    if dist.is_initialized(): # Only destroy if initialized
        dist.destroy_process_group()

# --- Training ---
def create_training_configs_sparsify(
        n_layers, checkpoint_dir,
        expansion_factor=32, lr=0.0004,
        train_batch_size = 2048,
        lr_warm_up_steps=5000,
        seed=42,
        log_to_wandb=False,
        save_every = 1000,
        loss_fn: Literal["ce", "fvu", "kl"] = "fvu",
        optimizer: Literal["adam", "muon", "signum"] = "signum",
        grad_acc_steps: int = 1,
        micro_acc_steps: int = 1,
        run_name: Optional[str] = None,
        wandb_log_frequency: int = 1
        ) -> Dict[int, TrainConfig]:
    # 1. Create the SparseCoderConfig (TranscoderConfig)
    transcoder_sae_cfg = TranscoderConfig(
        expansion_factor=expansion_factor,
    )

    # 2. Generate the list of hookpoints based on n_layers
    hookpoints = [f"blocks.{layer}.ln2.hook_normalized" for layer in range(n_layers)]
    layers = list(range(n_layers)) # Also include layers if you want to specify them explicitly

    # 3. Create the TrainConfig
    cfg = TrainConfig(
        sae=transcoder_sae_cfg, # Pass the transcoder_sae_cfg to the 'sae' field
        batch_size=train_batch_size,
        lr=lr,
        lr_warmup_steps=lr_warm_up_steps,
        log_to_wandb=log_to_wandb,
        save_dir=str(checkpoint_dir),
        init_seeds=[seed],
        hookpoints=hookpoints,
        layers=layers,
        save_every=save_every,
        loss_fn=loss_fn,
        optimizer=optimizer,
        grad_acc_steps=grad_acc_steps,
        micro_acc_steps=micro_acc_steps,
        run_name=run_name,
        wandb_log_frequency=wandb_log_frequency,
        # Other parameters from TrainConfig can be set here or left as defaults
    )

    return cfg


def train_worker(rank, world_size, model_name):
    """The worker function for training using sparsify with torchrun."""
    setup(rank, world_size)
    device = f"cuda:{rank}"
    torch.cuda.set_device(device) # Ensure this process uses the correct GPU

    checkpoint_dir = DEFAULT_PROJECT_PATH / "output" / model_name
    # Only create dir on main process to avoid race conditions if multiple processes try to create it
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Ensure all processes wait for directory creation
    if world_size > 1:
        dist.barrier()


    n_layers = get_n_layers(model_name)
    layer_to_configs = create_training_configs_sparsify(
        n_layers=n_layers,
        checkpoint_dir=checkpoint_dir,
        lr = 0.0004,
        )
    paths = []

    # Load model and tokenizer once per process
    # Explicitly move model to the correct device for this process
    repo_name = get_repo_name(model_name=model_name)
    model = AutoModelForCausalLM.from_pretrained(
        repo_name,
        torch_dtype=torch.bfloat16, # Or whatever dtype you chose in config
    ).to(device) # Move model to specific device here

    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize the dataset once per process
    # Sparsify's Trainer, when launched with torchrun, should handle distributed data loading
    # (e.g., through DistributedSampler implicitly or by shard logic)
    dataset = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train")
    tokenized_dataset = chunk_and_tokenize(dataset, tokenizer)

    for layer, cfg in layer_to_configs.items():
        if rank == 0:
            print(f"About to start training for layer {layer} with lr {cfg.lr}")
            print(f"Checkpoint path: {cfg.checkpoint_dir}")

        # Sparsify's Trainer manages the SAE creation and training
        # It's expected to handle DDP internally when run via torchrun
        trainer = Trainer(
            cfg,
            tokenized_dataset,
            model,
            # No need to pass rank/world_size explicitly to Trainer here,
            # as it will infer them from the torch.distributed environment.
        )

        trainer.fit()

        # Save SAE to checkpoints folder on the main process
        if rank == 0:
            final_sae_path = checkpoint_dir / f"final_sae_layer_{layer}.pt"
            trainer.save_model(final_sae_path)
            paths.append(str(final_sae_path))

    cleanup() # Call cleanup at the end of the worker
    return paths # Return paths only from rank 0 is meaningful for the user

# --- Argument Parsing and Main Execution ---

def get_repo_name(model_name: str) -> str:
    """Maps string argument to replacement model version"""
    supported_models = {
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it":  "google/gemma-2-2b-it",
        "gemma-2-9b" : "google/gemma-2-9b",
        "gemma-2-9b-it" : "google/gemma-2-9b-it",
        "gemma-2-27b" : "google/gemma-2-27b",
        "gemma-2-27b" : "google/gemma-2-27b-it",
        "shieldgemma-2b": "google/shieldgemma-2b",
        "shieldgemma-9b": "google/shieldgemma-9b",
        "gpt-2": "openai-community/gpt2",
    }
    if model_name not in supported_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(supported_models.keys())}")
    return supported_models[model_name]

def get_n_layers(model_name: str) -> int:
    supported_models = {
        "gemma-2-2b" : 25,
        "gemma-2-2b-it" : 25,
        "gemma-2-9b" : 41,
        "gemma-2-9b-it" : 41,
        'gemma-2-27b' : 45,
        "gemma-2-27b-it": 45,
        "shieldgemma-2b": 25,
        "shieldgemma-9b": 41,
        'gpt2': 11,
    }
    if model_name not in supported_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(supported_models.keys())}")
    return supported_models[model_name]

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Transcoders with DDP.")
    parser.add_argument(
        "--model_name", type=str, default='gemma-2-2b',
        choices=['gemma-2-2b', "gemma-2-2b-it", "gemma-2-9b", "gemma-2-9b-it",  "gemma-2-27b", "gemma-2-27b-it", "shieldgemma-2b", "shieldgemma-9b", "gpt2"],
        help="Model Name identifier (e.g., gemma-2-2b)."
    )
    # The --world_size argument is usually provided by torchrun as --nproc_per_node,
    # and then read from the environment by torch.distributed.
    # We keep it as a placeholder if you wanted to test with mp.spawn directly.
    parser.add_argument(
        "--world_size", type=int, default=1, # Default to 1 for local testing without torchrun
        help="Number of GPUs to use for training. (Set by torchrun via env vars)."
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to run the script."""
    setup_environment()

    args = parse_arguments()
    print(f"Training model {args.model_name} on {args.world_size} GPUs (if launched with torchrun).")

    # When using torchrun, it sets up the environment variables (LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    # and then executes your script. You don't call mp.spawn yourself within the script.
    # The 'rank' and 'world_size' will be available from the environment.
    current_rank = int(os.environ.get("RANK", 0))
    current_world_size = int(os.environ.get("WORLD_SIZE", 1))

    # The train_worker function will be called directly by torchrun for each process.
    # If not launched with torchrun, it will run as a single process (rank 0, world_size 1).
    train_worker(current_rank, current_world_size, args.model_name)


# torchrun --nproc_per_node 6 train_transcoder.py --model_name gemma-2-2b
if __name__ == "__main__":
    main()