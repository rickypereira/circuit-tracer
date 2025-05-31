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
import subprocess
import sys
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
DEFAULT_VM_INSTANCE_NAME = "circuit-tracking-us-west-1"

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
paths = setup_paths(Path(f"/home/ubuntu/{DEFAULT_VM_INSTANCE_NAME}").resolve())

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


def main():
    """Main function to run the script."""
    setup_environment()

    print("Loading model 'google/gemma-2-2b'...")
    # Note: This requires significant memory.
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        'gemma',
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
