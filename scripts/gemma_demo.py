#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemma-2-2b Demo - Executable Python Script

This script demonstrates attribution and intervention experiments on the Gemma-2-2b model.
It is a conversion of the original Jupyter Notebook demo.

The script performs several experiments, including:
- Swapping the language of a sentence from French to Spanish.
- Changing the subject of a sentence from basketball to football.
- Modifying the outcome of an arithmetic operation.
- Altering the language of a generated text from French to Spanish.

Prerequisites:
- Python 3.8+
- PyTorch
- Hugging Face Transformers and Hub
- The 'circuit-tracer' library

To install the necessary libraries, run:
    pip install torch transformers huggingface_hub
    pip install git+https://github.com/safety-research/circuit-tracer.git

You will also need to be authenticated with Hugging Face. You can do this by running:
    huggingface-cli login
"""

import os
import logging
import argparse
import sys
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
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

# Setup paths and environment
paths = setup_paths(DEFAULT_PROJECT_PATH)

from circuit_tracer.replacement_model import ReplacementModel
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model


# --- Utility Functions (re-implemented from the notebook's utils) ---

def print_topk_token_predictions(sentence, original_logits, new_logits, tokenizer, k=5):
    """
    Prints a comparison of the top-k token predictions before and after an intervention.
    """
    print("-" * 80)
    print(f"Input Sentence: '{sentence}'")
    print("-" * 80)

    # Process original logits
    print("\nOriginal Top 5 Tokens:")
    original_probs = torch.softmax(original_logits[0, -1], dim=-1)
    topk_probs, topk_indices = torch.topk(original_probs, k)
    for i in range(k):
        token = tokenizer.decode(topk_indices[i])
        prob = topk_probs[i].item()
        print(f"  {i+1}. '{token}' (Probability: {prob:.2%})")

    # Process new logits
    print("\nNew Top 5 Tokens (Post-intervention):")
    new_probs = torch.softmax(new_logits[0, -1], dim=-1)
    topk_probs, topk_indices = torch.topk(new_probs, k)
    for i in range(k):
        token = tokenizer.decode(topk_indices[i])
        prob = topk_probs[i].item()
        print(f"  {i+1}. '{token}' (Probability: {prob:.2%})")
    print("-" * 80 + "\n")


def print_generations_comparison(base_sentence, pre_gen, post_gen):
    """
    Prints a comparison of generated text before and after an intervention.
    """
    print("-" * 80)
    print("Generation Comparison")
    print("-" * 80)

    print("\nPre-intervention generations:")
    for i, gen in enumerate(pre_gen):
        # The generate function returns the full text including the prompt
        generated_text = gen[len(base_sentence):]
        print(f"  {i+1}. {base_sentence} --> '{generated_text}'")

    print("\nPost-intervention generations:")
    for i, gen in enumerate(post_gen):
        generated_text = gen[len(base_sentence):]
        print(f"  {i+1}. {base_sentence} --> '{generated_text}'")
    print("-" * 80 + "\n")


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


def run_language_and_sport_swap_experiments(model, print_topk_func):
    """
    Runs experiments for swapping languages (French->Spanish) and topics (Basketball->Football).
    """
    print("\n" + "="*30)
    print("Running Language & Sport Swap Experiments")
    print("="*30 + "\n")

    # --- Experiment 1: Change French prediction to Spanish ---
    # We want to change the output of "Fait: Michael Jordan joue au" (French)
    # from 'basket' to 'baloncesto' by using features from a Spanish sentence.
    s_french_mj = "Fait: Michael Jordan joue au"
    s_spanish_mj = "Hecho: Michael Jordan juega al"

    # Define features related to French and Spanish from the attribution graphs.
    french_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=20, pos=slice(6, 8), feature_idx=1454)
    spanish_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=20, pos=slice(6, 8), feature_idx=341)

    # Get activations from the Spanish sentence to apply to the French one.
    _, spanish_activations = model.get_activations(s_spanish_mj)
    interventions = [(*french_feature, 0), (*spanish_feature, 10 * spanish_activations[spanish_feature])]
    
    with torch.inference_mode():
        original_logits = model(s_french_mj)
        new_logits, _ = model.feature_intervention(s_french_mj, interventions)

    print_topk_func(s_french_mj, original_logits, new_logits)


    # --- Experiment 2: Change Basketball to Football ---
    # We change the predicted sport for Michael Jordan from basketball to football.
    s_football_brady = "Tom Brady plays the sport of"
    
    # Define features for football and basketball.
    football_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=16, pos=6, feature_idx=5039)
    basketball_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=18, pos=7, feature_idx=12672)

    _, football_activations = model.get_activations(s_football_brady)
    interventions = [(*basketball_feature, 0),
                     (football_feature.layer, basketball_feature.pos, football_feature.feature_idx, football_activations[football_feature])]

    with torch.inference_mode():
        original_logits = model(s_french_mj) # Using the French MJ sentence as base
        new_logits, _ = model.feature_intervention(s_french_mj, interventions)
    
    # Note: The original notebook shows a modest change here. We are just replicating.
    print_topk_func("Fait: Michael Jordan joue au [Basketball->Football Intervention]", original_logits, new_logits)


    # --- Experiment 3: Early-layer intervention for Basketball to Football ---
    # This intervention is done at an earlier layer.
    early_football_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=0, pos=2, feature_idx=1703)
    early_basketball_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=1, pos=5, feature_idx=1173)

    _, football_activations = model.get_activations(s_football_brady)
    interventions = [(*early_basketball_feature, 0),
                     (early_football_feature.layer, early_basketball_feature.pos, early_football_feature.feature_idx, 10 * football_activations[early_football_feature])]

    with torch.inference_mode():
        original_logits = model(s_french_mj)
        new_logits, _ = model.feature_intervention(s_french_mj, interventions)

    print_topk_func("Fait: Michael Jordan joue au [Early-Layer Basketball->Football Intervention]", original_logits, new_logits)


def run_analogy_experiments(model, print_topk_func):
    """
    Runs experiments on analogies, changing the relationship from language to currency.
    """
    print("\n" + "="*30)
    print("Running Analogy Experiments")
    print("="*30 + "\n")

    # --- Experiment 1: Change Country->Language to Country->Currency ---
    s_spanish_us = "Mexico:Spanish :: US:"
    s_peso_us = "Mexico:peso :: US:"

    # Features for "language" found in the "Mexico:Spanish" example.
    language_features = [
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=6, pos=3, feature_idx=1509),
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=9, pos=3, feature_idx=11486),
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=9, pos=3, feature_idx=16135),
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=14, pos=3, feature_idx=1107)
    ]
    
    # Features for "currency" found in the "Mexico:peso" example.
    currency_features = [
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=6, pos=3, feature_idx=2102),
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=9, pos=3, feature_idx=11294),
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=9, pos=3, feature_idx=13858),
        namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=14, pos=3, feature_idx=9246)
    ]

    _, currency_activations = model.get_activations(s_peso_us, sparse=True)

    # Swap language features for currency features
    interventions = [(*currency_feature, 10 * currency_activations[currency_feature]) for currency_feature in currency_features] + \
                    [(*language_feature, 0.0) for language_feature in language_features]

    with torch.inference_mode():
        original_logits = model(s_spanish_us)
        new_logits, _ = model.feature_intervention(s_spanish_us, interventions)

    print_topk_func(s_spanish_us, original_logits, new_logits)


def run_addition_experiment(model, print_topk_func):
    """
    Runs an experiment to change the result of an addition problem.
    """
    print("\n" + "="*30)
    print("Running Addition Experiment")
    print("="*30 + "\n")

    s3 = "2 + 1 = "
    s8 = "3 + 5 = "

    # Feature for the number '3' from the first equation
    feature3 = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=25, pos=6, feature_idx=10077)
    # Feature for the number '8' from the second equation
    feature8 = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=25, pos=6, feature_idx=14682)

    _, s8_activations = model.get_activations(s8, sparse=True)
    
    # Turn off the '3' feature and turn on the '8' feature
    interventions = [(*feature8, s8_activations[feature8]), (*feature3, 0)]
    
    with torch.inference_mode():
        original_logits = model(s3)
        # Here we intervene on s8 but display results for s3 to show the change
        new_logits, _ = model.feature_intervention(s3, interventions)

    print_topk_func(s3, original_logits, new_logits)


def run_generation_experiment(model, print_gen_func):
    """
    Runs experiments changing the language of multi-token generation.
    """
    print("\n" + "="*30)
    print("Running Generation Experiments")
    print("="*30 + "\n")

    # --- Experiment 1: Michael Jordan Sentence ---
    s_french_mj = "Fait: Michael Jordan joue au"
    s_spanish_mj = "Hecho: Michael Jordan juega al"
    
    french_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=20, pos=slice(6, 8), feature_idx=1454)
    spanish_feature = namedtuple('Feature', ['layer', 'pos', 'feature_idx'])(layer=20, pos=slice(6, 8), feature_idx=341)

    _, spanish_activations = model.get_activations(s_spanish_mj, sparse=False)
    
    interventions = [(spanish_feature.layer, slice(1, None), spanish_feature.feature_idx, 10 * spanish_activations[spanish_feature].mean()),
                     (french_feature.layer, slice(1, None), french_feature.feature_idx, 0)]
    
    hooks, _ = model._get_feature_intervention_hooks(s_french_mj, interventions, freeze_attention=False)
    
    with torch.inference_mode():
        pre_intervention_generation = [model.generate(s_french_mj, do_sample=False, use_past_kv_cache=False, verbose=False, max_new_tokens=15)]
        with model.hooks(hooks):
            post_intervention_generation = [model.generate(s_french_mj, do_sample=False, use_past_kv_cache=False, verbose=False, max_new_tokens=15)]
    
    print_gen_func(s_french_mj, pre_intervention_generation, post_intervention_generation)


    # --- Experiment 2: Seasons Sentence ---
    s_french_season = "La saison après le printemps s'appelle"
    s_spanish_season = "La estación después de la primavera se llama"

    # We can reuse the same language features if they are general enough
    _, spanish_activations = model.get_activations(s_spanish_season, sparse=False)
    
    hooks, _ = model._get_feature_intervention_hooks(s_french_season, interventions, freeze_attention=False)
    
    with torch.inference_mode():
        pre_intervention_generation = [model.generate(s_french_season, do_sample=False, use_past_kv_cache=False, verbose=False, max_new_tokens=15)]
        with model.hooks(hooks):
            post_intervention_generation = [model.generate(s_french_season, do_sample=False, use_past_kv_cache=False, verbose=False, max_new_tokens=15)]
            
    print_gen_func(s_french_season, pre_intervention_generation, post_intervention_generation)


# --- Training ---
def create_training_configs(
        model_name,  n_layers, checkpoint_path, n_checkpoints=3, d_in=768, d_out=728,
        expansion_factor=32, lr=0.0004, l1_coefficient=0.00014, b_dec_init_method='mean', 
        dataset_path="Skylion007/openwebtext", train_batch_size = 4096, 
        context_size = 128, lr_warm_up_steps=5000, n_batches_in_buffer = 128,
        total_training_tokens = 1_000_000 * 60, store_batch_size = 32, 
        use_ghost_grads=True, feature_sampling_method = None, feature_sampling_window = 1000,
        resample_batches=1028, dead_feature_window=5000, dead_feature_threshold = 1e-8,
        seed=42) -> Dict[int, LanguageModelSAERunnerConfig]:
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


def train(version: str) -> List[str]:
    checkpoint_path = DEFAULT_PROJECT_PATH / "output" / version
    n_layers = version_to_n_layers(version)
    layer_to_configs = create_training_configs(model_name=version, n_layers=n_layers, checkpoint_path=checkpoint_path)
    paths = []
    for _, cfg in layer_to_configs.items:
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

def map_version_arg(version_str: str) -> Tuple[str, str]:
    """Maps string argument to version enum."""
    version_map = {
        "gemma-2-2b": ("google/gemma-2-2b", "gemma"),
        "gemma-2-2b-it": ( "google/gemma-2-2b-it", "gemma"),
        "gemma-2-27b" : ("google/gemma-2-27b", "gemma"),
        "gemma-2-27b" : ("google/gemma-2-27b-it", "gemma"),
        "shieldgemma-9b": ("google/shieldgemma-9b", "gemma"),
        "gpt-2": ("openai-community/gpt2", "gpt"),
    }
    if version_str not in version_map:
        raise ValueError(f"Unknown version string: {version_str}. Available: {list(version_map.keys())}")
    return version_map[version_str]

def version_to_n_layers(version_str: str) -> int:
    version_map = {
        'gemma-2-27b' : 45
    }
    if version_str not in version_map:
        raise ValueError(f"Unknown version string: {version_str}. Available: {list(version_map.keys())}")
    return version_map[version_str]

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning and evaluation experiments.")
    # Model Arguments
    parser.add_argument(
        "--version", type=str, default='gemma-2-2b',
        choices=['gemma-2-2b', "gemma-2-27b"],
        help="Model version identifier (e.g., gemma-2-2b)."
    )
    parser.add_argument("--train", action='store_true', help="Run model training")
    args = parser.parse_args()
    return args


def main():
    """Main function to run the script."""
    setup_environment()

    args = parse_arguments()

    transcoder_set = None
    if args.train:
        print(f"Training model {args.version}")
        transcoder_set = train(version=args.version)
    
    if args.version == 'gemma-2-2b':
        transcoder_set = 'gemma'

    print(f"Loading model {args.version}")
    # Note: This requires significant memory.
    model = ReplacementModel.from_pretrained(
        model_name=map_version_arg(version=args.version),
        transcoder_set=transcoder_set,
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully.")

    # Create partial functions for display with the model's tokenizer
    print_topk_func = partial(print_topk_token_predictions, tokenizer=model.tokenizer)
    print_gen_func = partial(print_generations_comparison)
    
    # Run all experiments
    run_language_and_sport_swap_experiments(model, print_topk_func)
    run_analogy_experiments(model, print_topk_func)
    run_addition_experiment(model, print_topk_func)
    run_generation_experiment(model, print_gen_func)


if __name__ == "__main__":
    main()
