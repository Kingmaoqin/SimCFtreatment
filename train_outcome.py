"""
Training script for outcome model f_θ(S, C, T) -> Y.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np

from models import Encoder, OutcomeModel, InputProcessor


# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def train_outcome_epoch(
    outcome_model: OutcomeModel,
    encoder: Encoder,
    input_processor: InputProcessor,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train outcome model for one epoch.

    Returns:
        Average training loss
    """
    outcome_model.train()
    encoder.eval()  # Encoder is frozen
    input_processor.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        numeric = batch['numeric'].to(device)
        categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
        treatment = batch['treatment'].to(device)
        outcome = batch['outcome'].to(device)

        # Process input and encode (no gradients for encoder)
        with torch.no_grad():
            x = input_processor(numeric, categorical)
            S, C = encoder(x)

        # Predict outcome
        y_pred = outcome_model(S, C, treatment)

        # Compute loss
        loss = criterion(y_pred, outcome)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_outcome(
    outcome_model: OutcomeModel,
    encoder: Encoder,
    input_processor: InputProcessor,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate outcome model.

    Returns:
        Average validation loss, RMSE
    """
    outcome_model.eval()
    encoder.eval()
    input_processor.eval()

    total_loss = 0.0
    total_squared_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            numeric = batch['numeric'].to(device)
            categorical = {k: v.to(device) for k, v in batch['categorical'].items()}
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)

            # Process and encode
            x = input_processor(numeric, categorical)
            S, C = encoder(x)

            # Predict outcome
            y_pred = outcome_model(S, C, treatment)

            # Compute loss
            loss = criterion(y_pred, outcome)
            total_loss += loss.item()

            # Compute squared error for RMSE
            squared_error = ((y_pred - outcome) ** 2).sum().item()
            total_squared_error += squared_error
            num_samples += outcome.size(0)

    avg_loss = total_loss / len(val_loader)
    rmse = np.sqrt(total_squared_error / num_samples)

    return avg_loss, rmse


def train_outcome_model(
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
) -> OutcomeModel:
    """
    Train outcome model with frozen encoder.

    Returns:
        Trained outcome model
    """
    print("\n" + "="*60)
    print("Training Outcome Model f_θ(S, C, T) -> Y")
    print("="*60)

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    for param in input_processor.parameters():
        param.requires_grad = False

    # Initialize outcome model
    outcome_model = OutcomeModel(d_s, d_c, num_treatments).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(outcome_model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_outcome_epoch(
            outcome_model, encoder, input_processor,
            train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_rmse = validate_outcome(
            outcome_model, encoder, input_processor,
            val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss (MSE): {train_loss:.4f} | "
              f"Val Loss (MSE): {val_loss:.4f}, "
              f"Val RMSE: {val_rmse:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = outcome_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                outcome_model.load_state_dict(best_model_state)
                break

    print(f"\nBest validation MSE: {best_val_loss:.4f}")

    return outcome_model
