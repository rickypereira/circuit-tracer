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
from typing import Dict
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

    # Check for huggingface token via environment variable or then try HfApi().whoami()
    # The Hugging Face libraries will automatically use HF_TOKEN or HUGGING_FACE_HUB_TOKEN
    # if they are set. HfApi().whoami() will also use them if available.
    if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"):
        print("Hugging Face token found in environment variables.")
        try:
            from huggingface_hub.hf_api import HfApi
            HfApi().whoami() # This will use the token from env var if present
            print("Successfully authenticated with Hugging Face Hub.")
        except Exception as e:
            print(f"Error validating Hugging Face token from environment variables: {e}")
            print("Please ensure your Hugging Face token is valid.")
            sys.exit(1)
    else:
        print("Hugging Face token not found in environment variables.")
        print("Please set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable.")
        print("You can generate one here: https://huggingface.co/settings/tokens")
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


# --- Argument Parsing and Main Execution ---

def get_transcoder_set(model_name: str) -> str:
    NotImplemented

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

def hf_transcoder_set(model_name: str):
    supported_models = {
        "gemma-2-2b" : 'gemma'
    }
    if model_name not in supported_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(supported_models.keys())}")
    return supported_models[model_name]

def find_sae_safetensors_files(start_directory):
    """
    Collects all file paths that contain "sae.safetensors" within a given directory
    and its subdirectories.

    Args:
        start_directory (str): The directory to start the search from.

    Returns:
        list: A list of full file paths that contain "sae.safetensors".
              Returns an empty list if no such files are found or if the
              start_directory does not exist.
    """
    found_files = []
    target_filename = "sae.safetensors"

    if not os.path.isdir(start_directory):
        print(f"Error: The directory '{start_directory}' does not exist.")
        return []

    for root, _, files in os.walk(start_directory):
        for file in files:
            if file == target_filename:
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    return found_files

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Transcoders and Run Circuit Tracing.")
    # Model Arguments
    parser.add_argument(
        "--model_name", type=str, default='gemma-2-2b',
        choices=['gemma-2-2b', "gemma-2-27b"],
        help="Model Name identifier (e.g., gemma-2-2b)."
    )
    parser.add_argument(
        "--load_from_huggingface", action="store_true", help="Attempt to load transcoders from huggingface"
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to run the script."""
    setup_environment()

    args = parse_arguments()

    transcoder_set = None    
    if args.load_from_huggingface:
        transcoder_set = hf_transcoder_set(args.model_name)
    else:
        start_directory = DEFAULT_PROJECT_PATH / 'output' / args.model_name
        transcoder_set = find_sae_safetensors_files(start_directory)

    print(f"Loading model {args.model_name}")
    # Note: This requires significant memory.
    model = ReplacementModel.from_pretrained(
        model_name=get_repo_name(model_name=args.model_name),
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
