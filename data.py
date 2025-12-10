"""
Data loading, cleaning, and preprocessing for FCR-CD algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


# Define the intersection columns we need
REQUIRED_COLUMNS = [
    "RAW_ID", "UNIQUEID", "PROTOCOL", "VISIT",
    "THERAPY_STATUS", "THERAPY", "THERCODE",
    "AGE", "ORIGIN", "GENDER", "GEOCODE",
    "HAMD01", "HAMD02", "HAMD03", "HAMD04", "HAMD05", "HAMD06", "HAMD07",
    "HAMD08", "HAMD09", "HAMD10", "HAMD11", "HAMD12", "HAMD13", "HAMD14",
    "HAMD15", "HAMD16", "HAMD17"
]

HAMD_COLS = [f"HAMD{i:02d}" for i in range(1, 18)]


class PreprocessingConfig:
    """Stores preprocessing transformations fitted on training data."""

    def __init__(self):
        # Numeric feature stats
        self.numeric_mean: Dict[str, float] = {}
        self.numeric_std: Dict[str, float] = {}
        self.numeric_features: List[str] = []

        # Categorical vocabularies
        self.cat_vocabularies: Dict[str, Dict[str, int]] = {}
        self.cat_features: List[str] = []

        # Treatment vocabulary
        self.treatment_vocab: Dict[str, int] = {}
        self.treatment_inverse_vocab: Dict[int, str] = {}
        self.num_treatments: int = 0

        # Feature dimensions
        self.d_num: int = 0
        self.d_cat: int = 0


def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and perform initial cleaning.

    Args:
        csv_path: Path to CSV file

    Returns:
        Cleaned DataFrame with aligned schema
    """
    # Load with low_memory=False to avoid dtype warnings
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only intersection columns that exist in this file
    available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
    df = df[available_cols]

    # Enforce dtypes
    dtype_map = {
        'RAW_ID': 'int64',
        'VISIT': 'int64',
        'THERCODE': 'int64',
        'AGE': 'float32',
        'GEOCODE': 'str',
        'PROTOCOL': 'str',
        'UNIQUEID': 'str',
        'ORIGIN': 'str',
        'GENDER': 'str',
        'THERAPY_STATUS': 'str',
        'THERAPY': 'str'
    }

    for col in HAMD_COLS:
        if col in df.columns:
            dtype_map[col] = 'float32'

    # Apply dtype conversions
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype == 'str':
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert {col} to {dtype}: {e}")

    # Basic cleaning
    # Drop rows with null UNIQUEID or VISIT
    df = df.dropna(subset=['UNIQUEID', 'VISIT'])

    # Drop rows with null THERAPY or THERAPY_STATUS
    df = df.dropna(subset=['THERAPY', 'THERAPY_STATUS'])

    # Check HAMD items completeness
    valid_hamd = df[HAMD_COLS].notna().all(axis=1)
    df = df[valid_hamd].copy()

    # Remove duplicate visits (keep first)
    df = df.sort_values(['UNIQUEID', 'VISIT'])
    df = df.drop_duplicates(subset=['UNIQUEID', 'VISIT'], keep='first')

    # Add HAMD_TOTAL
    df['HAMD_TOTAL'] = df[HAMD_COLS].sum(axis=1)

    return df


def create_next_visit_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create samples where X and T come from visit v, and Y comes from visit v+1.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with one row per (patient, visit with next visit available)
    """
    samples = []

    for patient_id, group in df.groupby('UNIQUEID'):
        group = group.sort_values('VISIT').reset_index(drop=True)

        # For each visit except the last
        for i in range(len(group) - 1):
            current_visit = group.iloc[i]
            next_visit = group.iloc[i + 1]

            # Create sample
            sample = current_visit.copy()
            sample['NEXT_HAMD_TOTAL'] = next_visit['HAMD_TOTAL']
            samples.append(sample)

    return pd.DataFrame(samples).reset_index(drop=True)


def fit_preprocessing(df: pd.DataFrame) -> PreprocessingConfig:
    """
    Fit preprocessing transformations on training data.

    Args:
        df: Training DataFrame

    Returns:
        PreprocessingConfig with fitted transformations
    """
    config = PreprocessingConfig()

    # Define feature lists
    config.numeric_features = ['AGE', 'VISIT'] + HAMD_COLS + ['HAMD_TOTAL']
    config.cat_features = ['PROTOCOL', 'ORIGIN', 'GENDER', 'GEOCODE', 'THERAPY_STATUS']

    # Fit numeric statistics
    for col in config.numeric_features:
        if col in df.columns:
            config.numeric_mean[col] = float(df[col].mean())
            config.numeric_std[col] = float(df[col].std() + 1e-8)  # Add epsilon to avoid division by zero

    # Fit categorical vocabularies
    for col in config.cat_features:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            config.cat_vocabularies[col] = {val: idx for idx, val in enumerate(unique_vals)}

    # Fit treatment vocabulary
    unique_therapies = df['THERAPY'].dropna().unique()
    config.treatment_vocab = {therapy: idx for idx, therapy in enumerate(unique_therapies)}
    config.treatment_inverse_vocab = {idx: therapy for therapy, idx in config.treatment_vocab.items()}
    config.num_treatments = len(config.treatment_vocab)

    # Set dimensions
    config.d_num = len(config.numeric_features)
    config.d_cat = len(config.cat_features)

    return config


def apply_preprocessing(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    Apply preprocessing transformations to data.

    Args:
        df: DataFrame to transform
        config: Fitted PreprocessingConfig

    Returns:
        Transformed DataFrame
    """
    df = df.copy()

    # Standardize numeric features
    for col in config.numeric_features:
        if col in df.columns:
            df[col] = (df[col] - config.numeric_mean[col]) / config.numeric_std[col]

    # Encode categorical features
    for col in config.cat_features:
        if col in df.columns:
            # Map to integer, use -1 for unknown categories
            df[col + '_encoded'] = df[col].map(config.cat_vocabularies[col]).fillna(-1).astype(int)

    # Encode treatment
    df['THERAPY_encoded'] = df['THERAPY'].map(config.treatment_vocab).fillna(-1).astype(int)

    return df


class HAMDDataset(Dataset):
    """PyTorch Dataset for HAMD treatment effect data."""

    def __init__(self, df: pd.DataFrame, config: PreprocessingConfig):
        """
        Args:
            df: Preprocessed DataFrame with encoded features
            config: PreprocessingConfig with vocabularies and stats
        """
        self.config = config

        # Extract numeric features
        self.numeric_features = torch.tensor(
            df[config.numeric_features].values,
            dtype=torch.float32
        )

        # Extract categorical features
        self.cat_features = {}
        for col in config.cat_features:
            if col + '_encoded' in df.columns:
                self.cat_features[col] = torch.tensor(
                    df[col + '_encoded'].values,
                    dtype=torch.long
                )

        # Extract treatment labels
        self.treatment = torch.tensor(
            df['THERAPY_encoded'].values,
            dtype=torch.long
        )

        # Extract outcomes
        self.outcome = torch.tensor(
            df['NEXT_HAMD_TOTAL'].values,
            dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.numeric_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with keys:
                - 'numeric': numeric features
                - 'categorical': dict of categorical features
                - 'treatment': treatment label
                - 'outcome': outcome value
        """
        item = {
            'numeric': self.numeric_features[idx],
            'categorical': {name: feat[idx] for name, feat in self.cat_features.items()},
            'treatment': self.treatment[idx],
            'outcome': self.outcome[idx]
        }
        return item


def patient_level_split(df: pd.DataFrame,
                       train_size: float = 0.7,
                       val_size: float = 0.15,
                       test_size: float = 0.15,
                       seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data at patient level to avoid leakage.

    Args:
        df: DataFrame with patient data
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        seed: Random seed

    Returns:
        train_df, val_df, test_df
    """
    # Get unique patient IDs
    unique_patients = df['UNIQUEID'].unique()

    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        unique_patients,
        train_size=train_size,
        random_state=seed
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_ratio,
        random_state=seed
    )

    # Split dataframes
    train_df = df[df['UNIQUEID'].isin(train_patients)].copy()
    val_df = df[df['UNIQUEID'].isin(val_patients)].copy()
    test_df = df[df['UNIQUEID'].isin(test_patients)].copy()

    return train_df, val_df, test_df


def load_and_prepare_dataset(
    csv_path: str,
    batch_size: int = 64,
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader, PreprocessingConfig, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to load, clean, and prepare data.

    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for DataLoaders
        seed: Random seed
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion

    Returns:
        train_loader, val_loader, test_loader, preprocessing_config,
        train_df_processed, val_df_processed, test_df_processed
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading data from {csv_path}...")

    # Load and clean
    df = load_and_clean_csv(csv_path)
    print(f"Loaded {len(df)} visits from {df['UNIQUEID'].nunique()} patients")

    # Create next-visit samples
    df_samples = create_next_visit_samples(df)
    print(f"Created {len(df_samples)} next-visit samples")

    # Patient-level split
    train_df, val_df, test_df = patient_level_split(
        df_samples, train_size, val_size, test_size, seed
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Fit preprocessing on training data
    config = fit_preprocessing(train_df)
    print(f"Fitted preprocessing: {config.d_num} numeric features, "
          f"{config.d_cat} categorical features, {config.num_treatments} treatments")

    # Apply preprocessing
    train_df_processed = apply_preprocessing(train_df, config)
    val_df_processed = apply_preprocessing(val_df, config)
    test_df_processed = apply_preprocessing(test_df, config)

    # Create datasets
    train_dataset = HAMDDataset(train_df_processed, config)
    val_dataset = HAMDDataset(val_df_processed, config)
    test_dataset = HAMDDataset(test_df_processed, config)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, config, train_df_processed, val_df_processed, test_df_processed
