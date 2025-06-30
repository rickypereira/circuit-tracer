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
from contextlib import nullcontext, redirect_stdout
from datetime import timedelta

# --- Import from sparsify ---
from sparsify import TranscoderConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_PROJECT_PATH=Path(f"/home/rickpereira").resolve()
DEFAULT_WANDB_PROJECT = 'cloud-ai-research-experimental-runs'

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
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(weeks=1))
    torch.cuda.set_device(rank)

def cleanup():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
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
        wandb_log_frequency: int = 1,
        # New parameters from command line
        k: Optional[int] = None,
        layer_stride: Optional[int] = None,
        ctx_len: int = 2048,
        distribute_modules: bool = False,
        load_in_8bit: bool = False,
        ) -> TrainConfig:
    transcoder_sae_cfg = TranscoderConfig(
        expansion_factor=expansion_factor,
        k=k if k is not None else 0
    )

    layers = list(range(n_layers))
    if layer_stride is not None:
        layers = layers[::layer_stride]


    cfg = TrainConfig(
        sae=transcoder_sae_cfg,
        batch_size=train_batch_size,
        lr=lr,
        lr_warmup_steps=lr_warm_up_steps,
        log_to_wandb=log_to_wandb,
        save_dir=str(checkpoint_dir),
        init_seeds=[seed],
        layers=layers,
        save_every=save_every,
        loss_fn=loss_fn,
        optimizer=optimizer,
        grad_acc_steps=grad_acc_steps,
        micro_acc_steps=micro_acc_steps,
        run_name=run_name,
        wandb_log_frequency=wandb_log_frequency,
        distribute_modules=distribute_modules
    )
    return cfg


def train_worker(rank, world_size, args):
    """The worker function for training using sparsify with torchrun."""
    with nullcontext() if rank == 0 else redirect_stdout(None):
        setup(rank, world_size)
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        dataset_path = get_dataset_path(name=args.dataset_path)
        if dataset_path == '':
            print("Dataset Path is not supported")
            sys.exit(1)

        checkpoint_dir = DEFAULT_PROJECT_PATH / "output" / args.model_name
        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()


        n_layers = get_n_layers(args.model_name)
        sparsify_cfg = create_training_configs_sparsify(
            n_layers=n_layers,
            checkpoint_dir=checkpoint_dir,
            train_batch_size=args.batch_size,
            lr = 0.0004,
            log_to_wandb=args.log_to_wandb,
            grad_acc_steps=args.grad_acc_steps,
            micro_acc_steps=args.micro_acc_steps,
            k=args.k,
            layer_stride=args.layer_stride,
            ctx_len=args.ctx_len,
        )
        paths = []

        if args.load_in_8bit:
            dtype = torch.float16
            # For 8-bit loading, BitsAndBytesConfig will handle the compute_dtype
            nf8_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="int8",
                bnb_8bit_compute_dtype=torch.float16, 
                bnb_8bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                get_repo_name(args.model_name),
                quantization_config=nf8_config,
                device_map={"": f"cuda:{rank}"},
                torch_dtype=dtype,
            )
        else:
            # Determine dtype for model loading for non-8bit
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                get_repo_name(args.model_name),
                device_map={"": f"cuda:{rank}"},
                torch_dtype=dtype
            )

        tokenizer = AutoTokenizer.from_pretrained(get_repo_name(args.model_name))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load and tokenize the dataset once per process
        dataset = load_dataset(dataset_path, split="train")
        if args.dataset_train_size:
            print(f"Loading Dataset Train of Size {args.dataset_train_size}")
            dataset = dataset.select(range(args.dataset_train_size))

        tokenized_dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=args.ctx_len)

        trainer = Trainer(
            sparsify_cfg,
            tokenized_dataset,
            model,
        )

        trainer.fit()

        cleanup()
        return paths

# --- Argument Parsing and Main Execution ---

def get_repo_name(model_name: str) -> str:
    """Maps string argument to replacement model version"""
    supported_models = {
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it":  "google/gemma-2-2b-it",
        "gemma-2-9b" : "google/gemma-2-9b",
        "gemma-2-9b-it" : "google/gemma-2-9b-it",
        "gemma-2-27b" : "google/gemma-2-27b",
        "gemma-2-27b-it" : "google/gemma-2-27b-it",
        "shieldgemma-2b": "google/shieldgemma-2b",
        "shieldgemma-9b": "google/shieldgemma-9b",
        "gpt-2": "openai-community/gpt2",
    }
    if model_name not in supported_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(supported_models.keys())}")
    return supported_models[model_name]

def get_dataset_path(name: str) -> str:
    """Maps argument name to downloadable dataset name."""
    supported_datasets = {
        "openwebtext": "Skylion007/openwebtext",
        "smol": "EleutherAI/SmolLM2-135M-10B",
    }
    return supported_datasets.get(name.lower(), '')

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
    parser.add_argument(
        "--dataset_path", type=str, default='openwebtext',
        choices=['openwebtext', "smol"],
        help="The Dataset Path."
    )
    # Add new arguments
    parser.add_argument(
        "--distribute_modules", action="store_true",
        help="Whether to distribute modules across GPUs (handled by Sparsify Trainer's DDP)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, # This is per-GPU batch size
        help="Batch size per GPU."
    )
    parser.add_argument(
        "--layer_stride", type=int, default=1,
        help="Stride for selecting layers (e.g., 2 means every other layer)."
    )
    parser.add_argument(
        "--grad_acc_steps", type=int, default=1,
        help="Number of gradient accumulation steps."
    )
    parser.add_argument(
        "--ctx_len", type=int, default=2048,
        help="Context length for tokenization."
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Parameter 'k' for sparse autoencoder configuration (e.g., top-k selection)."
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true",
        help="Load the model in 8-bit quantization."
    )
    parser.add_argument(
        "--micro_acc_steps", type=int, default=1,
        help="Number of micro accumulation steps."
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true",
        help="Log to Weights & Biases."
    )
    parser.add_argument(
        "--dataset_train_size", type=int, default=None,
        help="Optional. The size of the dataset to be trained on."
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to run the script."""
    setup_environment()

    args = parse_arguments()
    current_rank = int(os.environ.get("RANK", 0))
    current_world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Training model {args.model_name} on rank {current_rank}/{current_world_size} GPUs.")
    print(f"Arguments: {args}")

    train_worker(current_rank, current_world_size, args)


# torchrun --nproc_per_node gpu -m scripts.train_transcoder --model_name gemma-2-2b --distribute_modules --batch_size 1 --layer_stride 2 --grad_acc_steps 8 --ctx_len 2048 --k 192 --load_in_8bit --micro_acc_steps 2
if __name__ == "__main__":
    main()