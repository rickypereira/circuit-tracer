# Transcoder training sample code

"""
This sample script can be used to train a transcoder on a model of your choice.
This code, along with the transcoder training code more generally, was largely
    adapted from an older version of Joseph Bloom's SAE training repo, the latest
    version of which can be found at https://github.com/jbloomAus/SAELens.
Most of the parameters given here are the same as the SAE training parameters
    listed at https://jbloomaus.github.io/SAELens/training_saes/.
Transcoder-specific parameters are marked as such in comments.

"""

import logging
import torch
import argparse
import os 
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_PROJECT_PATH=Path(f"/home/rickpereira").resolve()

# --- Helper Functions ---
def setup_paths(base_dir: Path) -> Dict[str, Path]:
    """Sets up project paths based on a base directory."""
    project_path = base_dir
    workspace_path = project_path / "circuit-tracer"
    output_dir = project_path / "output"

    if str(workspace_path) not in sys.path:
        logging.info(f"Adding {workspace_path} to sys.path")
        sys.path.append(str(workspace_path))

    logging.info(f"Setting MODEL_PATH environment variable to {project_path}")
    os.environ['MODEL_PATH'] = str(project_path)

    logging.info(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project": project_path,
        "workspace": workspace_path,
        "output": output_dir,
    }

def setup_environment():
    """
    Handles environment setup. In the original notebook, this installed a git repository.
    For this script, we assume the user has installed the library via pip as per
    the instructions in the docstring.
    """
    print("Ensuring environment is set up...")
    try:
        import circuit_tracer
        print("'circuit-tracer' library found.")
    except ImportError:
        print("Error: 'circuit-tracer' library not found.")
        print("Please install it by running: pip install git+https://github.com/safety-research/circuit-tracer.git")
        sys.exit(1)

    try:
        # Check for huggingface login
        from huggingface_hub.hf_api import HfApi
        HfApi().whoami()
        print("Hugging Face token found.")
    except Exception:
        print("Hugging Face token not found.")
        print("Please log in using: huggingface-cli login")
        sys.exit(1)

# Setup paths and environment
paths = setup_paths(DEFAULT_PROJECT_PATH)

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model


# --- Training ---
def create_training_configs(
        model_name,  n_layers, checkpoint_path, n_checkpoints=3, d_in=4608, d_out=4608,
        expansion_factor=32, lr=0.0004, l1_coefficient=0.00014, b_dec_init_method='mean', 
        dataset_path="Skylion007/openwebtext", train_batch_size = 4096, 
        context_size = 32, lr_warm_up_steps=5000, n_batches_in_buffer = 32,
        total_training_tokens = 1_000_000 * 60, store_batch_size = 16, 
        use_ghost_grads=True, feature_sampling_method = None, feature_sampling_window = 1000,
        resample_batches=1028, dead_feature_window=5000, dead_feature_threshold = 1e-8,
        seed=42, dtype=torch.float32) -> Dict[int, LanguageModelSAERunnerConfig]:
    layer_to_config = {}
    for layer in range(n_layers):
        cfg = LanguageModelSAERunnerConfig(
            hook_point = f"blocks.{layer}.ln2.hook_normalized",
            hook_point_layer = layer,
            d_in = d_in,
            dataset_path = dataset_path,
            is_dataset_tokenized=False,
            model_name=model_name,
            is_transcoder = True,
            out_hook_point = f"blocks.{layer}.hook_mlp_out",
            out_hook_point_layer = layer,
            d_out = d_out,
            # SAE Parameters
            expansion_factor = expansion_factor,
            b_dec_init_method = b_dec_init_method,
            
            # Training Parameters
            lr = lr,
            l1_coefficient = l1_coefficient,
            lr_scheduler_name="constantwithwarmup",
            train_batch_size = train_batch_size,
            context_size = context_size,
            lr_warm_up_steps=lr_warm_up_steps,
            
            # Activation Store Parameters
            n_batches_in_buffer = n_batches_in_buffer,
            total_training_tokens = total_training_tokens,
            store_batch_size = store_batch_size,
            
            # Dead Neurons and Sparsity
            use_ghost_grads=use_ghost_grads,
            feature_sampling_method = feature_sampling_method,
            feature_sampling_window = feature_sampling_window,
            resample_batches=resample_batches,
            dead_feature_window=dead_feature_window,
            dead_feature_threshold = dead_feature_threshold,

            # WANDB
            log_to_wandb = False,
            
            # Misc
            use_tqdm = True,
            device = "cuda",
            seed = seed,
            n_checkpoints = n_checkpoints,
            checkpoint_path = checkpoint_path,
            dtype = torch.float32,
        )
        layer_to_config[layer] = cfg
    return layer_to_config


def train(model_name: str) -> List[str]:
    checkpoint_path = DEFAULT_PROJECT_PATH / "output" / model_name
    n_layers = get_n_layers(model_name)
    layer_to_configs = create_training_configs(
        model_name=model_name, 
        n_layers=n_layers, 
        checkpoint_path=checkpoint_path,
        lr = 0.0004, l1_coefficient = 0.00014
        )
    paths = []
    for _, cfg in layer_to_configs.items():
        print(f"About to start training with lr {cfg.lr} and l1 {cfg.l1_coefficient}")
        print(f"Checkpoint path: {cfg.checkpoint_path}")
        print(cfg)
        loader = LMSparseAutoencoderSessionloader(cfg)
        model, sparse_autoencoder, activations_loader = loader.load_session()
        # train SAE
        sparse_autoencoder = train_sae_on_language_model(
            model, sparse_autoencoder, activations_loader,
            n_checkpoints=cfg.n_checkpoints,
            batch_size = cfg.train_batch_size,
            feature_sampling_method = cfg.feature_sampling_method,
            feature_sampling_window = cfg.feature_sampling_window,
            feature_reinit_scale = cfg.feature_reinit_scale,
            dead_feature_threshold = cfg.dead_feature_threshold,
            dead_feature_window=cfg.dead_feature_window,
            use_wandb = cfg.log_to_wandb,
            wandb_log_frequency = cfg.wandb_log_frequency
        )

        # save sae to checkpoints folder
        path = f"{cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
        sparse_autoencoder.save_model(path)
        paths.append(path)
    return paths
        
# --- Argument Parsing and Main Execution ---

def get_n_layers(model_name: str) -> int:
    supported_models = {
        'gemma-2-27b' : 45,
        'gpt2': 11,
    }
    if model_name not in supported_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(supported_models.keys())}")
    return supported_models[model_name]

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Transcoders.")
    # Model Arguments
    parser.add_argument(
        "--model_name", type=str, default='gemma-2-2b',
        choices=['gemma-2-2b', "gemma-2-27b", "gpt2"],
        help="Model Name identifier (e.g., gemma-2-2b)."
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to run the script."""
    setup_environment()

    args = parse_arguments()
    print(f"Training model {args.model_name}")
    paths = train(model_name=args.model_name)
    for path in paths:
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()