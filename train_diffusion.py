"""
Training script for conditional diffusion model on confounding representation C.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np

from models import Encoder, DiffusionModel, InputProcessor


# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Diffusion parameters
NUM_TIMESTEPS = 100  # T
BETA_START = 1e-4
BETA_END = 0.02


def get_diffusion_schedule(num_timesteps: int, beta_start: float, beta_end: float) -> Dict[str, torch.Tensor]:
    """
    Create diffusion schedule with precomputed values.

    Returns:
        Dictionary containing beta, alpha, alpha_bar tensors
    """
    # Linear schedule for beta
    betas = torch.linspace(beta_start, beta_end, num_timesteps)

    # Compute alphas
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars
    }


def diffusion_forward_sample(
    c_0: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from forward diffusion process q(c_t | c_0).

    c_t = sqrt(alpha_bar_t) * c_0 + sqrt(1 - alpha_bar_t) * eps

    Args:
        c_0: Clean confounding representations [batch_size, d_c]
        t: Timesteps [batch_size] (integer indices)
        alpha_bars: Precomputed alpha_bar values [num_timesteps]
        device: Device

    Returns:
        c_t: Noisy samples [batch_size, d_c]
        eps: Sampled noise [batch_size, d_c]
    """
    batch_size, d_c = c_0.shape

    # Get alpha_bar for each timestep
    # If alpha_bars and t are on different devices, use CPU indexing
    if alpha_bars.device != t.device:
        t_cpu = t.cpu()
        alpha_bar_t = alpha_bars[t_cpu].view(-1, 1).to(device)
    else:
        # Both on same device, can index directly
        alpha_bar_t = alpha_bars[t].view(-1, 1)

    # Sample noise
    eps = torch.randn_like(c_0)

    # Forward process
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

    c_t = sqrt_alpha_bar_t * c_0 + sqrt_one_minus_alpha_bar_t * eps

    return c_t, eps


def train_diffusion_epoch(
    diffusion_model: DiffusionModel,
    encoder: Encoder,
    input_processor: InputProcessor,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    diffusion_schedule: Dict[str, torch.Tensor],
    device: torch.device
) -> float:
    """
    Train diffusion model for one epoch.

    Returns:
        Average training loss
    """
    diffusion_model.train()
    encoder.eval()  # Encoder is frozen
    input_processor.eval()

    total_loss = 0.0
    num_batches = 0

    # Ensure alpha_bars is on the correct device for better performance
    alpha_bars = diffusion_schedule['alpha_bars'].to(device)

    for batch in train_loader:
        numeric = batch['numeric'].to(device)
        categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
        treatment = batch['treatment'].to(device)

        batch_size = numeric.shape[0]

        # Get representations (no gradients for encoder)
        with torch.no_grad():
            x = input_processor(numeric, categorical)
            S, C = encoder(x)

        # Sample random timesteps for each sample in batch
        t = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=device)

        # Forward diffusion to get noisy C
        c_t, eps = diffusion_forward_sample(C, t, alpha_bars, device)

        # Predict noise
        eps_pred = diffusion_model(c_t, t.float(), S, treatment)

        # Compute loss
        loss = nn.MSELoss()(eps_pred, eps)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_diffusion(
    diffusion_model: DiffusionModel,
    encoder: Encoder,
    input_processor: InputProcessor,
    val_loader: DataLoader,
    diffusion_schedule: Dict[str, torch.Tensor],
    device: torch.device
) -> float:
    """
    Validate diffusion model.

    Returns:
        Average validation loss
    """
    diffusion_model.eval()
    encoder.eval()
    input_processor.eval()

    total_loss = 0.0
    num_batches = 0

    # Ensure alpha_bars is on the correct device for better performance
    alpha_bars = diffusion_schedule['alpha_bars'].to(device)

    with torch.no_grad():
        for batch in val_loader:
            numeric = batch['numeric'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            treatment = batch['treatment'].to(device)

            batch_size = numeric.shape[0]

            # Get representations
            x = input_processor(numeric, categorical)
            S, C = encoder(x)

            # Sample random timesteps
            t = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=device)

            # Forward diffusion
            c_t, eps = diffusion_forward_sample(C, t, alpha_bars, device)

            # Predict noise
            eps_pred = diffusion_model(c_t, t.float(), S, treatment)

            # Compute loss
            loss = nn.MSELoss()(eps_pred, eps)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_diffusion_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder: Encoder,
    input_processor: InputProcessor,
    num_treatments: int,
    d_s: int,
    d_c: int,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE
) -> Tuple[DiffusionModel, Dict[str, torch.Tensor]]:
    """
    Train conditional diffusion model on C.

    Returns:
        Trained diffusion model and diffusion schedule
    """
    print("\n" + "="*60)
    print("Training Diffusion Model ε_ψ(c_t, t, S, T)")
    print("="*60)

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    for param in input_processor.parameters():
        param.requires_grad = False

    # Create diffusion schedule
    diffusion_schedule = get_diffusion_schedule(NUM_TIMESTEPS, BETA_START, BETA_END)
    print(f"Diffusion timesteps: {NUM_TIMESTEPS}, beta: [{BETA_START}, {BETA_END}]")

    # Initialize diffusion model
    diffusion_model = DiffusionModel(d_c, d_s, num_treatments).to(device)

    # Optimizer
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_diffusion_epoch(
            diffusion_model, encoder, input_processor,
            train_loader, optimizer, diffusion_schedule, device
        )

        # Validate
        val_loss = validate_diffusion(
            diffusion_model, encoder, input_processor,
            val_loader, diffusion_schedule, device
        )

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = diffusion_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                diffusion_model.load_state_dict(best_model_state)
                break

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return diffusion_model, diffusion_schedule
