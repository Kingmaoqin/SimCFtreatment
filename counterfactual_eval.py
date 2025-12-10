"""
Counterfactual generation and treatment effect estimation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np

from models import Encoder, OutcomeModel, DiffusionModel, InputProcessor


def reverse_diffusion_sample(
    diffusion_model: DiffusionModel,
    S: torch.Tensor,
    T_target: torch.Tensor,
    diffusion_schedule: Dict[str, torch.Tensor],
    d_c: int,
    device: torch.device,
    num_samples: int = 1
) -> torch.Tensor:
    """
    Generate counterfactual C via reverse diffusion conditioned on S and target treatment T_target.

    Args:
        diffusion_model: Trained diffusion model
        S: Stable representation [batch_size, d_s]
        T_target: Target treatment labels [batch_size]
        diffusion_schedule: Dictionary with diffusion parameters
        d_c: Confounding dimension
        device: Device
        num_samples: Number of samples to generate per input

    Returns:
        Counterfactual C samples [batch_size, num_samples, d_c]
    """
    diffusion_model.eval()

    batch_size = S.shape[0]
    betas = diffusion_schedule['betas'].to(device)
    alphas = diffusion_schedule['alphas'].to(device)
    alpha_bars = diffusion_schedule['alpha_bars'].to(device)

    num_timesteps = len(betas)

    # Store all samples
    all_samples = []

    for _ in range(num_samples):
        # Start from pure noise
        c_t = torch.randn(batch_size, d_c, device=device)

        # Reverse process: t = T-1, T-2, ..., 0
        for t_idx in reversed(range(num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.float32)

            # Predict noise
            with torch.no_grad():
                eps_pred = diffusion_model(c_t, t, S, T_target)

            # Get parameters
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]

            # Compute mean
            # μ_t = 1/sqrt(α_t) * (c_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_pred)
            coeff1 = 1.0 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (c_t - coeff2 * eps_pred)

            if t_idx > 0:
                # Add noise (not on last step)
                # Variance: σ_t^2 = β_t
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(c_t)
                c_t = mean + sigma_t * z
            else:
                # Final step: no noise
                c_t = mean

        all_samples.append(c_t.unsqueeze(1))  # [batch_size, 1, d_c]

    # Concatenate samples
    c_counterfactual = torch.cat(all_samples, dim=1)  # [batch_size, num_samples, d_c]

    return c_counterfactual


def generate_counterfactual_predictions(
    encoder: Encoder,
    outcome_model: OutcomeModel,
    diffusion_model: DiffusionModel,
    input_processor: InputProcessor,
    data_loader: DataLoader,
    diffusion_schedule: Dict[str, torch.Tensor],
    target_treatment: int,
    d_s: int,
    d_c: int,
    device: torch.device,
    num_diffusion_samples: int = 5
) -> Dict[str, np.ndarray]:
    """
    Generate counterfactual predictions for all samples in data_loader.

    Args:
        encoder: Trained encoder
        outcome_model: Trained outcome model
        diffusion_model: Trained diffusion model
        input_processor: Input processor
        data_loader: DataLoader with samples
        diffusion_schedule: Diffusion parameters
        target_treatment: Integer index of target treatment
        d_s: Stable dimension
        d_c: Confounding dimension
        device: Device
        num_diffusion_samples: Number of diffusion samples to average

    Returns:
        Dictionary with:
            - 'factual_outcome': True outcomes
            - 'predicted_factual': Predicted factual outcomes
            - 'predicted_counterfactual': Predicted counterfactual outcomes
            - 'treatment': Observed treatments
            - 'ite': Individual treatment effects
    """
    encoder.eval()
    outcome_model.eval()
    diffusion_model.eval()
    input_processor.eval()

    all_factual_outcomes = []
    all_predicted_factual = []
    all_predicted_counterfactual = []
    all_treatments = []

    with torch.no_grad():
        for batch in data_loader:
            numeric = batch['numeric'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)

            batch_size = numeric.shape[0]

            # Get representations
            x = input_processor(numeric, categorical)
            S, C = encoder(x)

            # Predict factual outcome
            y_factual_pred = outcome_model(S, C, treatment)

            # Generate counterfactual C for target treatment
            T_target = torch.full_like(treatment, target_treatment)
            C_cf_samples = reverse_diffusion_sample(
                diffusion_model, S, T_target,
                diffusion_schedule, d_c, device,
                num_samples=num_diffusion_samples
            )  # [batch_size, num_samples, d_c]

            # Predict counterfactual outcomes (average over samples)
            y_cf_preds = []
            for sample_idx in range(num_diffusion_samples):
                C_cf = C_cf_samples[:, sample_idx, :]  # [batch_size, d_c]
                y_cf = outcome_model(S, C_cf, T_target)
                y_cf_preds.append(y_cf)

            y_cf_pred = torch.stack(y_cf_preds).mean(dim=0)  # [batch_size]

            # Store results
            all_factual_outcomes.append(outcome.cpu().numpy())
            all_predicted_factual.append(y_factual_pred.cpu().numpy())
            all_predicted_counterfactual.append(y_cf_pred.cpu().numpy())
            all_treatments.append(treatment.cpu().numpy())

    # Concatenate all batches
    factual_outcomes = np.concatenate(all_factual_outcomes)
    predicted_factual = np.concatenate(all_predicted_factual)
    predicted_counterfactual = np.concatenate(all_predicted_counterfactual)
    treatments = np.concatenate(all_treatments)

    # Compute individual treatment effects
    ite = predicted_counterfactual - predicted_factual

    return {
        'factual_outcome': factual_outcomes,
        'predicted_factual': predicted_factual,
        'predicted_counterfactual': predicted_counterfactual,
        'treatment': treatments,
        'ite': ite
    }


def compute_ate(
    encoder: Encoder,
    outcome_model: OutcomeModel,
    diffusion_model: DiffusionModel,
    input_processor: InputProcessor,
    data_loader: DataLoader,
    diffusion_schedule: Dict[str, torch.Tensor],
    baseline_treatment: int,
    target_treatment: int,
    d_s: int,
    d_c: int,
    device: torch.device,
    num_diffusion_samples: int = 5
) -> float:
    """
    Compute Average Treatment Effect (ATE) between baseline and target treatment.

    ATE(target, baseline) = E[Y(target) - Y(baseline)]

    Args:
        encoder: Trained encoder
        outcome_model: Trained outcome model
        diffusion_model: Trained diffusion model
        input_processor: Input processor
        data_loader: DataLoader with samples
        diffusion_schedule: Diffusion parameters
        baseline_treatment: Baseline treatment index
        target_treatment: Target treatment index
        d_s: Stable dimension
        d_c: Confounding dimension
        device: Device
        num_diffusion_samples: Number of diffusion samples

    Returns:
        ATE estimate
    """
    encoder.eval()
    outcome_model.eval()
    diffusion_model.eval()
    input_processor.eval()

    all_y_baseline = []
    all_y_target = []

    with torch.no_grad():
        for batch in data_loader:
            numeric = batch['numeric'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}

            batch_size = numeric.shape[0]

            # Get representations
            x = input_processor(numeric, categorical)
            S, C = encoder(x)

            # Generate C for baseline treatment
            T_baseline = torch.full((batch_size,), baseline_treatment, device=device, dtype=torch.long)
            C_baseline_samples = reverse_diffusion_sample(
                diffusion_model, S, T_baseline,
                diffusion_schedule, d_c, device,
                num_samples=num_diffusion_samples
            )

            # Predict outcome under baseline
            y_baseline_preds = []
            for sample_idx in range(num_diffusion_samples):
                C_bl = C_baseline_samples[:, sample_idx, :]
                y_bl = outcome_model(S, C_bl, T_baseline)
                y_baseline_preds.append(y_bl)
            y_baseline = torch.stack(y_baseline_preds).mean(dim=0)

            # Generate C for target treatment
            T_target = torch.full((batch_size,), target_treatment, device=device, dtype=torch.long)
            C_target_samples = reverse_diffusion_sample(
                diffusion_model, S, T_target,
                diffusion_schedule, d_c, device,
                num_samples=num_diffusion_samples
            )

            # Predict outcome under target
            y_target_preds = []
            for sample_idx in range(num_diffusion_samples):
                C_tg = C_target_samples[:, sample_idx, :]
                y_tg = outcome_model(S, C_tg, T_target)
                y_target_preds.append(y_tg)
            y_target = torch.stack(y_target_preds).mean(dim=0)

            all_y_baseline.append(y_baseline.cpu().numpy())
            all_y_target.append(y_target.cpu().numpy())

    # Concatenate and compute ATE
    y_baseline_all = np.concatenate(all_y_baseline)
    y_target_all = np.concatenate(all_y_target)

    ate = np.mean(y_target_all - y_baseline_all)

    return ate


def evaluate_counterfactuals(
    encoder: Encoder,
    outcome_model: OutcomeModel,
    diffusion_model: DiffusionModel,
    input_processor: InputProcessor,
    test_loader: DataLoader,
    diffusion_schedule: Dict[str, torch.Tensor],
    treatment_vocab_inverse: Dict[int, str],
    d_s: int,
    d_c: int,
    device: torch.device,
    num_display_samples: int = 10
) -> Dict:
    """
    Evaluate and display counterfactual predictions.

    Args:
        encoder: Trained encoder
        outcome_model: Trained outcome model
        diffusion_model: Trained diffusion model
        input_processor: Input processor
        test_loader: Test DataLoader
        diffusion_schedule: Diffusion parameters
        treatment_vocab_inverse: Mapping from treatment index to name
        d_s: Stable dimension
        d_c: Confounding dimension
        device: Device
        num_display_samples: Number of samples to display

    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*60)
    print("Counterfactual Evaluation")
    print("="*60)

    # Find most common treatment as target
    all_treatments = []
    for batch in test_loader:
        all_treatments.extend(batch['treatment'].numpy())
    all_treatments = np.array(all_treatments)

    unique, counts = np.unique(all_treatments, return_counts=True)
    most_common_treatment_idx = unique[np.argmax(counts)]
    most_common_treatment_name = treatment_vocab_inverse[most_common_treatment_idx]

    print(f"\nMost common treatment: {most_common_treatment_name} (index: {most_common_treatment_idx})")
    print(f"Using this as target treatment for counterfactual generation.\n")

    # Generate counterfactuals
    results = generate_counterfactual_predictions(
        encoder, outcome_model, diffusion_model, input_processor,
        test_loader, diffusion_schedule,
        target_treatment=most_common_treatment_idx,
        d_s=d_s, d_c=d_c, device=device
    )

    # Display some examples
    print(f"Sample Counterfactual Predictions (first {num_display_samples}):")
    print("-" * 80)
    print(f"{'Obs Treatment':<20} {'True Y':<12} {'Pred Y (fact)':<15} {'Pred Y (cf)':<15} {'ITE':<10}")
    print("-" * 80)

    for i in range(min(num_display_samples, len(results['factual_outcome']))):
        obs_treatment_idx = results['treatment'][i]
        obs_treatment_name = treatment_vocab_inverse.get(obs_treatment_idx, f"Unknown({obs_treatment_idx})")
        true_y = results['factual_outcome'][i]
        pred_y_fact = results['predicted_factual'][i]
        pred_y_cf = results['predicted_counterfactual'][i]
        ite = results['ite'][i]

        print(f"{obs_treatment_name:<20} {true_y:<12.3f} {pred_y_fact:<15.3f} {pred_y_cf:<15.3f} {ite:<10.3f}")

    print("-" * 80)

    # Overall statistics
    mean_ite = np.mean(results['ite'])
    std_ite = np.std(results['ite'])
    print(f"\nOverall ITE Statistics:")
    print(f"  Mean ITE: {mean_ite:.4f}")
    print(f"  Std ITE:  {std_ite:.4f}")

    # Compute ATE between two treatments
    if len(unique) >= 2:
        # Pick two most common treatments
        sorted_indices = np.argsort(-counts)
        baseline_idx = unique[sorted_indices[0]]
        target_idx = unique[sorted_indices[1]] if len(unique) > 1 else unique[0]

        baseline_name = treatment_vocab_inverse[baseline_idx]
        target_name = treatment_vocab_inverse[target_idx]

        print(f"\nComputing ATE between {baseline_name} (baseline) and {target_name} (target)...")

        ate = compute_ate(
            encoder, outcome_model, diffusion_model, input_processor,
            test_loader, diffusion_schedule,
            baseline_treatment=baseline_idx,
            target_treatment=target_idx,
            d_s=d_s, d_c=d_c, device=device
        )

        print(f"ATE({target_name}, {baseline_name}) = {ate:.4f}")
        print(f"(Positive means {target_name} leads to higher outcome on average)")
    else:
        ate = None
        baseline_idx = None
        target_idx = None
        baseline_name = None
        target_name = None

    # Return detailed results
    return {
        'ite_mean': mean_ite,
        'ite_std': std_ite,
        'ite_values': results['ite'],
        'ate': ate,
        'ate_baseline': baseline_name,
        'ate_target': target_name,
        'ate_baseline_idx': baseline_idx,
        'ate_target_idx': target_idx,
        'n_test_samples': len(results['ite']),
        'treatment_counts': dict(zip(unique, counts))
    }
