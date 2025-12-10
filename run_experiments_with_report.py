"""
Run experiments with multiple seeds and generate comprehensive evaluation report.

Usage:
    python run_experiments_with_report.py --csv_path treatment_switched.csv --experiment_name switched --seeds 42 43 44
"""

import argparse
import random
import numpy as np
import torch
import os
import sys

from data import load_and_prepare_dataset
from train_representation import train_representation_model, D_S, D_C
from train_outcome import train_outcome_model
from train_diffusion import train_diffusion_model
from counterfactual_eval import evaluate_counterfactuals, compute_ate
from results_reporter import ResultsReporter


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_seed_experiment(
    csv_path: str,
    experiment_name: str,
    seed: int,
    device: torch.device,
    reporter: ResultsReporter
):
    """
    Run experiment for a single seed.

    Args:
        csv_path: Path to CSV data file
        experiment_name: Name for this experiment
        seed: Random seed
        device: PyTorch device
        reporter: ResultsReporter instance
    """
    print("\n" + "="*80)
    print(f"Running Seed: {seed}")
    print("="*80)

    # Set seeds
    set_all_seeds(seed)

    # Load data
    print(f"\nLoading data from {csv_path}...")
    train_loader, val_loader, test_loader, config, train_df, val_df, test_df = load_and_prepare_dataset(
        csv_path=csv_path,
        batch_size=64,
        seed=seed
    )

    print(f"Dataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Treatments: {config.num_treatments}")

    # Get categorical vocabulary sizes
    cat_vocab_sizes = {name: len(vocab) for name, vocab in config.cat_vocabularies.items()}

    # Train representation model
    print(f"\n[Seed {seed}] Training representation model...")
    encoder, propensity_head, input_processor = train_representation_model(
        train_loader=train_loader,
        val_loader=val_loader,
        d_num=config.d_num,
        cat_vocab_sizes=cat_vocab_sizes,
        num_treatments=config.num_treatments,
        device=device
    )

    # Train outcome model
    print(f"\n[Seed {seed}] Training outcome model...")
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
    print(f"\n[Seed {seed}] Evaluating on test set...")
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

    print(f"[Seed {seed}] Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

    # Train diffusion model
    print(f"\n[Seed {seed}] Training diffusion model...")
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

    # Counterfactual evaluation
    print(f"\n[Seed {seed}] Counterfactual evaluation...")
    eval_results = evaluate_counterfactuals(
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
        num_display_samples=10
    )

    # Add results to reporter
    reporter.add_seed_result(
        seed=seed,
        metrics={
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'ite_mean': eval_results['ite_mean'],
            'ite_std': eval_results['ite_std'],
            'n_test_samples': eval_results['n_test_samples']
        }
    )

    # Add ATE comparison to reporter
    if eval_results['ate'] is not None:
        reporter.add_ate_comparison(
            seed=seed,
            baseline=eval_results['ate_baseline'],
            target=eval_results['ate_target'],
            ate=eval_results['ate'],
            n_samples=eval_results['n_test_samples']
        )

    # Compute additional ATE comparisons for top treatments
    treatment_counts = eval_results['treatment_counts']
    sorted_treatments = sorted(treatment_counts.items(), key=lambda x: x[1], reverse=True)

    # Compare top 3 treatments if available
    for i in range(min(3, len(sorted_treatments))):
        for j in range(i + 1, min(3, len(sorted_treatments))):
            baseline_idx = sorted_treatments[i][0]
            target_idx = sorted_treatments[j][0]

            baseline_name = config.treatment_inverse_vocab[baseline_idx]
            target_name = config.treatment_inverse_vocab[target_idx]

            # Skip if this is the pair we already computed
            if (baseline_name == eval_results['ate_baseline'] and
                target_name == eval_results['ate_target']):
                continue

            print(f"\n[Seed {seed}] Computing ATE: {target_name} vs {baseline_name}...")

            ate = compute_ate(
                encoder, outcome_model, diffusion_model, input_processor,
                test_loader, diffusion_schedule,
                baseline_treatment=baseline_idx,
                target_treatment=target_idx,
                d_s=D_S, d_c=D_C, device=device
            )

            print(f"ATE = {ate:.4f}")

            reporter.add_ate_comparison(
                seed=seed,
                baseline=baseline_name,
                target=target_name,
                ate=ate,
                n_samples=eval_results['n_test_samples']
            )

    print(f"\n[Seed {seed}] ✓ Completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run FCR-CD experiments with multiple seeds and generate report"
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
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 43, 44],
        help='List of random seeds to use (default: 42 43 44)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for reports'
    )

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize reporter
    reporter = ResultsReporter(args.experiment_name, args.output_dir)

    # Run experiments for each seed
    for seed in args.seeds:
        try:
            run_single_seed_experiment(
                csv_path=args.csv_path,
                experiment_name=args.experiment_name,
                seed=seed,
                device=device,
                reporter=reporter
            )
        except Exception as e:
            print(f"\n❌ Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate and save report
    print("\n" + "="*80)
    print("Generating Final Report")
    print("="*80)

    report_path = reporter.save_report()

    print("\n" + "="*80)
    print("✓ ALL EXPERIMENTS COMPLETED")
    print(f"✓ Report saved to: {report_path}")
    print("="*80)


if __name__ == '__main__':
    main()
