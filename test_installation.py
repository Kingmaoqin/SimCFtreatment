"""
Test script to verify installation and data loading.

Run this after installing dependencies:
    pip install -r requirements.txt
    python test_installation.py
"""

import sys
import torch
import numpy as np
import pandas as pd

print("="*60)
print("Testing Installation and Setup")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from data import load_and_prepare_dataset
    from models import Encoder, PropensityHead, OutcomeModel, DiffusionModel, InputProcessor
    from train_representation import train_representation_model
    from train_outcome import train_outcome_model
    from train_diffusion import train_diffusion_model
    from counterfactual_eval import evaluate_counterfactuals
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test data loading
print("\n2. Testing data loading on treatment_switched.csv...")
try:
    train_loader, val_loader, test_loader, config, _, _, _ = load_and_prepare_dataset(
        csv_path='treatment_switched.csv',
        batch_size=64,
        seed=42
    )
    print(f"   ✓ Data loaded successfully")
    print(f"     - Train batches: {len(train_loader)}")
    print(f"     - Val batches: {len(val_loader)}")
    print(f"     - Test batches: {len(test_loader)}")
    print(f"     - Num treatments: {config.num_treatments}")
except Exception as e:
    print(f"   ✗ Data loading error: {e}")
    sys.exit(1)

# Test batch structure
print("\n3. Testing batch structure...")
try:
    batch = next(iter(train_loader))
    print(f"   ✓ Batch loaded successfully")
    print(f"     - Numeric features shape: {batch['numeric'].shape}")
    print(f"     - Treatment shape: {batch['treatment'].shape}")
    print(f"     - Outcome shape: {batch['outcome'].shape}")
    print(f"     - Categorical features: {list(batch['categorical'].keys())}")
except Exception as e:
    print(f"   ✗ Batch loading error: {e}")
    sys.exit(1)

# Test model instantiation
print("\n4. Testing model instantiation...")
try:
    device = torch.device('cpu')
    d_s, d_c = 32, 16

    # Get vocab sizes
    cat_vocab_sizes = {name: len(vocab) for name, vocab in config.cat_vocabularies.items()}

    # Input processor
    input_processor = InputProcessor(config.d_num, cat_vocab_sizes).to(device)
    d_in = input_processor.d_out

    # Encoder
    encoder = Encoder(d_in, d_s, d_c).to(device)

    # Propensity head
    propensity_head = PropensityHead(d_c, config.num_treatments).to(device)

    # Outcome model
    outcome_model = OutcomeModel(d_s, d_c, config.num_treatments).to(device)

    # Diffusion model
    diffusion_model = DiffusionModel(d_c, d_s, config.num_treatments).to(device)

    print("   ✓ All models instantiated successfully")
except Exception as e:
    print(f"   ✗ Model instantiation error: {e}")
    sys.exit(1)

# Test forward pass
print("\n5. Testing forward pass...")
try:
    batch_cpu = {
        'numeric': batch['numeric'].to(device),
        'categorical': {k: v.to(device) for k, v in batch['categorical'].items()},
        'treatment': batch['treatment'].to(device),
        'outcome': batch['outcome'].to(device)
    }

    # Input processing
    x = input_processor(batch_cpu['numeric'], batch_cpu['categorical'])

    # Encoding
    S, C = encoder(x)

    # Propensity prediction
    logits = propensity_head(C)

    # Outcome prediction
    y_pred = outcome_model(S, C, batch_cpu['treatment'])

    # Diffusion (noise prediction)
    t = torch.randint(0, 100, (batch_cpu['numeric'].shape[0],), device=device).float()
    eps_pred = diffusion_model(C, t, S, batch_cpu['treatment'])

    print("   ✓ Forward pass successful")
    print(f"     - S shape: {S.shape}")
    print(f"     - C shape: {C.shape}")
    print(f"     - Propensity logits shape: {logits.shape}")
    print(f"     - Outcome prediction shape: {y_pred.shape}")
    print(f"     - Diffusion noise prediction shape: {eps_pred.shape}")
except Exception as e:
    print(f"   ✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Ready to run experiments.")
print("="*60)
print("\nTo run experiments:")
print("  python main.py --csv_path treatment_switched.csv --experiment_name switched --seed 42")
print("  python main.py --csv_path treatment_consistent.csv --experiment_name consistent --seed 42")
print("\nOr run both:")
print("  python run_all.py")
print()
