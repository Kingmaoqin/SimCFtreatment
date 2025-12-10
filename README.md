# SimCFtreatment: FCR-CD Algorithm Implementation

Complete implementation of the **Factorized Causal Representation + Confounding Diffusion (FCR-CD)** algorithm for causal treatment effect estimation from observational data.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python test_installation.py
```

### 3. Run Experiments

**Single experiment:**
```bash
python main.py --csv_path treatment_switched.csv --experiment_name switched --seed 42
```

**Both datasets:**
```bash
python run_all.py
```

## What This Algorithm Does

The FCR-CD algorithm learns to estimate individual and average treatment effects by:

1. **Factorizing** patient features into stable (S) and confounding (C) representations
2. **Modeling** treatment propensity from confounders
3. **Predicting** outcomes from representations and treatment
4. **Generating** counterfactual scenarios using conditional diffusion
5. **Estimating** what would happen under different treatments

## Files

- `data.py` - Data loading, cleaning, preprocessing
- `models.py` - Neural network architectures (Encoder, Propensity, Outcome, Diffusion)
- `train_representation.py` - Self-supervised learning for representations
- `train_outcome.py` - Outcome model training
- `train_diffusion.py` - Conditional diffusion model training
- `counterfactual_eval.py` - Counterfactual generation and evaluation
- `main.py` - Main experiment orchestration
- `run_all.py` - Run both experiments sequentially
- `test_installation.py` - Installation verification

## Documentation

See `ALGORITHM_README.md` for detailed algorithm description, hyperparameters, and technical details.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- CPU-only (no GPU required)

## Output

Each experiment produces:
- Training progress for all model components
- Test set performance (MSE, RMSE)
- Counterfactual predictions for sample patients
- Individual Treatment Effect (ITE) statistics
- Average Treatment Effect (ATE) estimates
- Saved models in `models_{experiment_name}/`