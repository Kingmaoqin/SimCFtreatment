"""
Training script for representation learning (encoder + propensity head) with SSL losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from models import Encoder, PropensityHead, InputProcessor, add_gaussian_noise, apply_feature_dropout


# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Representation dimensions
D_S = 32  # Stable representation dimension
D_C = 16  # Confounding representation dimension

# Loss weights
LAMBDA_S = 1.0          # Stability loss weight
LAMBDA_PC = 0.5         # Propensity consistency weight
LAMBDA_PF = 1.0         # Propensity fitting weight
LAMBDA_DEC = 0.1        # Decorrelation weight
LAMBDA_VAR = 0.01       # Variance regularization weight

# Augmentation parameters
NOISE_STD = 0.01
DROPOUT_P = 0.05

# Variance threshold
VAR_THRESHOLD = 1e-3


def compute_stability_loss(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Compute stability loss: mean squared distance between S from two augmented views.

    L_S = mean_i || S_i^{(1)} - S_i^{(2)} ||_2^2
    """
    return torch.mean((S1 - S2) ** 2)


def compute_propensity_consistency_loss(P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
    """
    Compute propensity consistency loss.

    L_prop_cons = mean_i || P_i^{(1)} - P_i^{(2)} ||_2^2
    """
    return torch.mean((P1 - P2) ** 2)


def compute_decorrelation_loss(S: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Compute decorrelation loss between S and C.

    L_decouple = || Cov(S, C) ||_F^2
    """
    batch_size = S.shape[0]

    # Center the representations
    S_centered = S - S.mean(dim=0, keepdim=True)
    C_centered = C - C.mean(dim=0, keepdim=True)

    # Cross-covariance matrix
    cov_SC = torch.mm(S_centered.T, C_centered) / batch_size

    # Frobenius norm squared
    return torch.sum(cov_SC ** 2)


def compute_variance_regularization(S: torch.Tensor, C: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Encourage variance in each dimension to be above threshold.

    L_var = sum_k max(0, v - std(S[:,k]))^2 + sum_k max(0, v - std(C[:,k]))^2
    """
    S_std = torch.std(S, dim=0)
    C_std = torch.std(C, dim=0)

    S_var_loss = torch.sum(torch.relu(threshold - S_std) ** 2)
    C_var_loss = torch.sum(torch.relu(threshold - C_std) ** 2)

    return S_var_loss + C_var_loss


def train_representation_epoch(
    encoder: Encoder,
    propensity_head: PropensityHead,
    input_processor: InputProcessor,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary of loss components
    """
    encoder.train()
    propensity_head.train()
    input_processor.train()

    total_losses = {
        'total': 0.0,
        'stability': 0.0,
        'prop_cons': 0.0,
        'prop_fit': 0.0,
        'decorr': 0.0,
        'var': 0.0
    }

    num_batches = 0

    for batch in train_loader:
        numeric = batch['numeric'].to(device)
        categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
        treatment = batch['treatment'].to(device)

        # Process input
        x = input_processor(numeric, categorical)

        # Create two augmented views
        x1 = add_gaussian_noise(x, NOISE_STD)
        x1 = apply_feature_dropout(x1, DROPOUT_P)

        x2 = add_gaussian_noise(x, NOISE_STD)
        x2 = apply_feature_dropout(x2, DROPOUT_P)

        # Encode both views
        S1, C1 = encoder(x1)
        S2, C2 = encoder(x2)

        # (1) Stability loss
        loss_S = compute_stability_loss(S1, S2)

        # (2) Propensity consistency loss
        logits1 = propensity_head(C1)
        logits2 = propensity_head(C2)
        P1 = torch.softmax(logits1, dim=-1)
        P2 = torch.softmax(logits2, dim=-1)
        loss_prop_cons = compute_propensity_consistency_loss(P1, P2)

        # (3) Propensity fitting loss (on non-augmented input)
        S, C = encoder(x)
        logits = propensity_head(C)
        loss_prop_fit = nn.CrossEntropyLoss()(logits, treatment)

        # (4) Decorrelation loss
        loss_decorr = compute_decorrelation_loss(S, C)

        # (5) Variance regularization
        loss_var = compute_variance_regularization(S, C, VAR_THRESHOLD)

        # Total SSL loss
        loss_ssl = (LAMBDA_S * loss_S +
                   LAMBDA_PC * loss_prop_cons +
                   LAMBDA_PF * loss_prop_fit +
                   LAMBDA_DEC * loss_decorr +
                   LAMBDA_VAR * loss_var)

        # Backward pass
        optimizer.zero_grad()
        loss_ssl.backward()
        optimizer.step()

        # Accumulate losses
        total_losses['total'] += loss_ssl.item()
        total_losses['stability'] += loss_S.item()
        total_losses['prop_cons'] += loss_prop_cons.item()
        total_losses['prop_fit'] += loss_prop_fit.item()
        total_losses['decorr'] += loss_decorr.item()
        total_losses['var'] += loss_var.item()

        num_batches += 1

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def validate_representation(
    encoder: Encoder,
    propensity_head: PropensityHead,
    input_processor: InputProcessor,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate representation model.

    Returns:
        Dictionary of validation metrics
    """
    encoder.eval()
    propensity_head.eval()
    input_processor.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            numeric = batch['numeric'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            treatment = batch['treatment'].to(device)

            # Process input
            x = input_processor(numeric, categorical)

            # Encode
            S, C = encoder(x)

            # Propensity prediction
            logits = propensity_head(C)
            loss = nn.CrossEntropyLoss()(logits, treatment)

            total_loss += loss.item()

            # Accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == treatment).sum().item()
            total += treatment.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0

    return {'loss': avg_loss, 'accuracy': accuracy}


def train_representation_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    d_num: int,
    cat_vocab_sizes: Dict[str, int],
    num_treatments: int,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE
) -> Tuple[Encoder, PropensityHead, InputProcessor]:
    """
    Train representation model (encoder + propensity head).

    Returns:
        Trained encoder, propensity_head, and input_processor
    """
    print("\n" + "="*60)
    print("Training Representation Model (Encoder + Propensity Head)")
    print("="*60)

    # Initialize models
    input_processor = InputProcessor(d_num, cat_vocab_sizes).to(device)
    d_in = input_processor.d_out

    encoder = Encoder(d_in, D_S, D_C).to(device)
    propensity_head = PropensityHead(D_C, num_treatments).to(device)

    # Optimizer
    params = list(encoder.parameters()) + list(propensity_head.parameters()) + list(input_processor.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_losses = train_representation_epoch(
            encoder, propensity_head, input_processor,
            train_loader, optimizer, device
        )

        # Validate
        val_metrics = validate_representation(
            encoder, propensity_head, input_processor,
            val_loader, device
        )

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses['total']:.4f} "
              f"(S: {train_losses['stability']:.4f}, "
              f"PC: {train_losses['prop_cons']:.4f}, "
              f"PF: {train_losses['prop_fit']:.4f}, "
              f"Dec: {train_losses['decorr']:.4f}, "
              f"Var: {train_losses['var']:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    return encoder, propensity_head, input_processor
