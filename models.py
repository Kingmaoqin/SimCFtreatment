"""
Neural network models for FCR-CD algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


class Encoder(nn.Module):
    """
    Encoder network that maps input features to stable (S) and confounding (C) representations.

    Φ_φ : R^{d_in} -> R^{d_s + d_c}
    """

    def __init__(self,
                 d_in: int,
                 d_s: int,
                 d_c: int,
                 hidden_dims: list = [256, 128]):
        """
        Args:
            d_in: Input dimension (after embedding concatenation)
            d_s: Stable representation dimension
            d_c: Confounding representation dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.d_s = d_s
        self.d_c = d_c

        layers = []
        prev_dim = d_in

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer splits into S and C
        layers.append(nn.Linear(prev_dim, d_s + d_c))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, d_in]

        Returns:
            S: Stable representation [batch_size, d_s]
            C: Confounding representation [batch_size, d_c]
        """
        out = self.network(x)
        S = out[:, :self.d_s]
        C = out[:, self.d_s:]
        return S, C


class PropensityHead(nn.Module):
    """
    Propensity head that predicts treatment probability from confounding representation.

    π_β(C) -> probability over treatments
    """

    def __init__(self, d_c: int, num_treatments: int, hidden_dim: int = 64):
        """
        Args:
            d_c: Confounding representation dimension
            num_treatments: Number of treatment categories
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_c, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_treatments)
        )

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: Confounding representation [batch_size, d_c]

        Returns:
            Logits over treatments [batch_size, num_treatments]
        """
        return self.network(C)


class OutcomeModel(nn.Module):
    """
    Outcome model that predicts outcome from stable and confounding representations + treatment.

    f_θ(S, C, T) -> Y
    """

    def __init__(self,
                 d_s: int,
                 d_c: int,
                 num_treatments: int,
                 treatment_embed_dim: int = 16,
                 hidden_dims: list = [128, 64]):
        """
        Args:
            d_s: Stable representation dimension
            d_c: Confounding representation dimension
            num_treatments: Number of treatment categories
            treatment_embed_dim: Treatment embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        # Treatment embedding
        self.treatment_embedding = nn.Embedding(num_treatments, treatment_embed_dim)

        # MLP for outcome prediction
        layers = []
        prev_dim = d_s + d_c + treatment_embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output: single outcome value

        self.network = nn.Sequential(*layers)

    def forward(self, S: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: Stable representation [batch_size, d_s]
            C: Confounding representation [batch_size, d_c]
            T: Treatment labels [batch_size]

        Returns:
            Predicted outcomes [batch_size]
        """
        T_embed = self.treatment_embedding(T)  # [batch_size, treatment_embed_dim]
        x = torch.cat([S, C, T_embed], dim=-1)  # [batch_size, d_s + d_c + treatment_embed_dim]
        out = self.network(x).squeeze(-1)  # [batch_size]
        return out


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [batch_size] (values in range [0, T])

        Returns:
            Time embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DiffusionModel(nn.Module):
    """
    Conditional diffusion model for confounding representation.

    ε_ψ(c_t, t, cond) -> predicted noise

    where cond = concat(S, treatment_embedding, time_embedding)
    """

    def __init__(self,
                 d_c: int,
                 d_s: int,
                 num_treatments: int,
                 time_embed_dim: int = 32,
                 treatment_embed_dim: int = 16,
                 hidden_dims: list = [128, 128, 128]):
        """
        Args:
            d_c: Confounding representation dimension
            d_s: Stable representation dimension
            num_treatments: Number of treatment categories
            time_embed_dim: Time embedding dimension
            treatment_embed_dim: Treatment embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        self.d_c = d_c

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)

        # Treatment embedding
        self.treatment_embedding = nn.Embedding(num_treatments, treatment_embed_dim)

        # Noise prediction network
        # Input: c_t (d_c) + S (d_s) + treatment_embed (treatment_embed_dim) + time_embed (time_embed_dim)
        input_dim = d_c + d_s + treatment_embed_dim + time_embed_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output: predicted noise (same dimension as c_t)
        layers.append(nn.Linear(prev_dim, d_c))

        self.network = nn.Sequential(*layers)

    def forward(self,
                c_t: torch.Tensor,
                t: torch.Tensor,
                S: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_t: Noisy confounding representation at timestep t [batch_size, d_c]
            t: Timesteps [batch_size]
            S: Stable representation [batch_size, d_s]
            T: Treatment labels [batch_size]

        Returns:
            Predicted noise [batch_size, d_c]
        """
        # Embed time and treatment
        t_embed = self.time_embedding(t)  # [batch_size, time_embed_dim]
        T_embed = self.treatment_embedding(T)  # [batch_size, treatment_embed_dim]

        # Concatenate all inputs
        x = torch.cat([c_t, S, T_embed, t_embed], dim=-1)

        # Predict noise
        eps_pred = self.network(x)

        return eps_pred


class InputProcessor(nn.Module):
    """
    Processes raw numeric and categorical features into a single vector for the encoder.
    """

    def __init__(self,
                 d_num: int,
                 cat_vocab_sizes: Dict[str, int],
                 cat_embed_dim: int = 8):
        """
        Args:
            d_num: Number of numeric features
            cat_vocab_sizes: Dictionary mapping categorical feature name to vocabulary size
            cat_embed_dim: Embedding dimension for each categorical feature
        """
        super().__init__()

        self.d_num = d_num

        # Create embeddings for categorical features
        self.cat_embeddings = nn.ModuleDict()
        self.cat_features = list(cat_vocab_sizes.keys())

        for name, vocab_size in cat_vocab_sizes.items():
            # Add 1 for unknown category (-1 -> 0 after offset)
            self.cat_embeddings[name] = nn.Embedding(vocab_size + 1, cat_embed_dim, padding_idx=0)

        self.d_cat_total = len(self.cat_features) * cat_embed_dim
        self.d_out = d_num + self.d_cat_total

    def forward(self, numeric: torch.Tensor, categorical: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            numeric: Numeric features [batch_size, d_num]
            categorical: Dict of categorical features [batch_size]

        Returns:
            Concatenated feature vector [batch_size, d_out]
        """
        # Start with numeric features
        features = [numeric]

        # Add embedded categorical features
        for name in self.cat_features:
            if name in categorical:
                cat_idx = categorical[name] + 1  # Offset by 1 (so -1 becomes 0, the padding idx)
                cat_embed = self.cat_embeddings[name](cat_idx)
                features.append(cat_embed)

        # Concatenate all
        return torch.cat(features, dim=-1)


def add_gaussian_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise for data augmentation."""
    noise = torch.randn_like(x) * std
    return x + noise


def apply_feature_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Apply dropout to features (not during training of main network)."""
    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    return x * mask
