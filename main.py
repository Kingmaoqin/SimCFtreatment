"""
Main script to orchestrate the full FCR-CD experimental pipeline.

Usage:
    python main.py --csv_path treatment_switched.csv --experiment_name switched --seed 42
    python main.py --csv_path treatment_consistent.csv --experiment_name consistent --seed 42
"""

import argparse
import random
import numpy as np
import torch
import os

from data import load_and_prepare_dataset
from train_representation import train_representation_model, D_S, D_C
from train_outcome import train_outcome_model
from train_diffusion import train_diffusion_model
from counterfactual_eval import evaluate_counterfactuals


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(csv_path: str, experiment_name: str, seed: int = 42):
    """
    Run full experiment pipeline.

    Args:
        csv_path: Path to CSV data file
        experiment_name: Name for this experiment
        seed: Random seed for reproducibility
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Data: {csv_path}")
    print(f"Seed: {seed}")
    print("="*80 + "\n")

    # Set seeds
    set_all_seeds(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ========================================================================
    # STEP 1: Load and prepare dataset
    # ========================================================================
    print("STEP 1: Loading and preparing dataset...")
    train_loader, val_loader, test_loader, config, train_df, val_df, test_df = load_and_prepare_dataset(
        csv_path=csv_path,
        batch_size=64,
        seed=seed
    )

    print(f"\nDataset statistics:")
    print(f"  Training samples:   {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples:       {len(test_df)}")
    print(f"  Number of treatments: {config.num_treatments}")
    print(f"  Numeric features: {config.d_num}")
    print(f"  Categorical features: {config.d_cat}")

    # Get categorical vocabulary sizes
    cat_vocab_sizes = {name: len(vocab) for name, vocab in config.cat_vocabularies.items()}

    # ========================================================================
    # STEP 2: Train representation model (Encoder + Propensity Head)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Training Representation Model")
    print("="*80)

    encoder, propensity_head, input_processor = train_representation_model(
        train_loader=train_loader,
        val_loader=val_loader,
        d_num=config.d_num,
        cat_vocab_sizes=cat_vocab_sizes,
        num_treatments=config.num_treatments,
        device=device
    )

    # ========================================================================
    # STEP 3: Train outcome model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Training Outcome Model")
    print("="*80)

    outcome_model = train_outcome_model(
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        input_processor=input_processor,
        num_treatments=config.num_treatments,
        d_s=D_S,
        d_c=D_C,
        device=device
    )

    # Evaluate on test set
    print("\nEvaluating outcome model on test set...")
    outcome_model.eval()
    encoder.eval()
    input_processor.eval()

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            numeric = batch['numeric'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)

            x = input_processor(numeric, categorical)
            S, C = encoder(x)
            y_pred = outcome_model(S, C, treatment)

            test_predictions.append(y_pred.cpu().numpy())
            test_targets.append(outcome.cpu().numpy())

    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)

    test_mse = np.mean((test_predictions - test_targets) ** 2)
    test_rmse = np.sqrt(test_mse)

    print(f"\n{'='*60}")
    print(f"Test Set Performance:")
    print(f"  MSE:  {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"{'='*60}")

    # ========================================================================
    # STEP 4: Train diffusion model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Training Diffusion Model")
    print("="*80)

    diffusion_model, diffusion_schedule = train_diffusion_model(
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        input_processor=input_processor,
        num_treatments=config.num_treatments,
        d_s=D_S,
        d_c=D_C,
        device=device
    )

    # ========================================================================
    # STEP 5: Counterfactual evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Counterfactual Generation and Evaluation")
    print("="*80)

    evaluate_counterfactuals(
        encoder=encoder,
        outcome_model=outcome_model,
        diffusion_model=diffusion_model,
        input_processor=input_processor,
        test_loader=test_loader,
        diffusion_schedule=diffusion_schedule,
        treatment_vocab_inverse=config.treatment_inverse_vocab,
        d_s=D_S,
        d_c=D_C,
        device=device,
        num_display_samples=15
    )

    # ========================================================================
    # STEP 6: Save models (optional)
    # ========================================================================
    save_dir = f"models_{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(encoder.state_dict(), f"{save_dir}/encoder.pt")
    torch.save(propensity_head.state_dict(), f"{save_dir}/propensity_head.pt")
    torch.save(outcome_model.state_dict(), f"{save_dir}/outcome_model.pt")
    torch.save(diffusion_model.state_dict(), f"{save_dir}/diffusion_model.pt")
    torch.save(input_processor.state_dict(), f"{save_dir}/input_processor.pt")

    print(f"\n✓ Models saved to {save_dir}/")

    print("\n" + "="*80)
    print(f"EXPERIMENT {experiment_name} COMPLETED")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FCR-CD: Factorized Causal Representation + Confounding Diffusion"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='Name for this experiment (e.g., "switched" or "consistent")'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Run experiment
    run_experiment(
        csv_path=args.csv_path,
        experiment_name=args.experiment_name,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
