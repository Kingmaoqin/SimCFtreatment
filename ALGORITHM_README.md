# FCR-CD: Factorized Causal Representation + Confounding Diffusion

This repository implements a complete experimental pipeline for causal treatment effect estimation using factorized causal representations and conditional diffusion models.

## Algorithm Overview

The FCR-CD algorithm learns to estimate individual and average treatment effects from observational data by:

1. **Factorized Representation Learning**: Decomposing patient features into:
   - **S (Stable)**: Treatment-invariant patient characteristics
   - **C (Confounding)**: Treatment-dependent confounding factors

2. **Propensity Modeling**: Predicting treatment assignment from confounders using π_β(C)

3. **Outcome Prediction**: Learning f_θ(S, C, T) → Y to predict outcomes

4. **Conditional Diffusion**: Using DDPM to generate counterfactual confounders C given alternative treatments

5. **Counterfactual Inference**: Estimating what would have happened under different treatments

## Project Structure

```
.
├── data.py                     # Data loading, cleaning, preprocessing
├── models.py                   # Neural network architectures
├── train_representation.py     # SSL training for encoder + propensity
├── train_outcome.py            # Outcome model training
├── train_diffusion.py          # Diffusion model training
├── counterfactual_eval.py      # Counterfactual generation and evaluation
├── main.py                     # Main experiment orchestration
├── run_all.py                  # Run both experiments sequentially
├── requirements.txt            # Python dependencies
├── treatment_switched.csv      # Dataset where treatment changes
└── treatment_consistent.csv    # Dataset where treatment stays the same
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Single Experiment

```bash
# Treatment switched dataset
python main.py --csv_path treatment_switched.csv --experiment_name switched --seed 42

# Treatment consistent dataset
python main.py --csv_path treatment_consistent.csv --experiment_name consistent --seed 42
```

### Run Both Experiments

```bash
python run_all.py
```

## Algorithm Details

### Phase 1: Representation Learning

The encoder Φ_φ maps input features X to (S, C) using a multi-component SSL loss:

```
L_SSL = λ_S * L_stability          # S should be invariant to augmentations
      + λ_PC * L_prop_consistency  # P(T|C) should be consistent
      + λ_PF * L_prop_fit          # P(T|C) should match observed treatments
      + λ_DEC * L_decorrelation    # S and C should be decorrelated
      + λ_VAR * L_variance         # Each dimension should have variance
```

**Key Components:**

- **Stability Loss**: Ensures S is stable across augmented views
- **Propensity Consistency**: Treatment probability from C should be consistent
- **Propensity Fitting**: Matches actual treatment assignments
- **Decorrelation**: Ensures S and C capture different information
- **Variance Regularization**: Prevents dimensional collapse

### Phase 2: Outcome Modeling

With frozen encoder, train f_θ(S, C, T) → Y using MSE loss:

```
L_outcome = MSE(f_θ(S, C, T), Y)
```

### Phase 3: Conditional Diffusion

Train a DDPM-style diffusion model ε_ψ(c_t, t, S, T) to model P(C | S, T):

**Forward Process:**
```
c_t = √(ᾱ_t) * c_0 + √(1 - ᾱ_t) * ε
```

**Training Objective:**
```
L_diffusion = E[||ε - ε_ψ(c_t, t, S, T)||²]
```

**Reverse Process** (generates counterfactual C):
```
For t = T, T-1, ..., 1:
    ε_pred = ε_ψ(c_t, t, S, T_counterfactual)
    c_{t-1} = reverse_step(c_t, ε_pred, t)
```

### Phase 4: Counterfactual Inference

For a patient with representation (S, C) under observed treatment T:

1. Generate counterfactual confounder C_cf via reverse diffusion conditioned on (S, T')
2. Predict factual outcome: ŷ(T) = f_θ(S, C, T)
3. Predict counterfactual outcome: ŷ(T') = f_θ(S, C_cf, T')
4. Estimate ITE: τ = ŷ(T') - ŷ(T)
5. Estimate ATE: average ITE across population

## Model Hyperparameters

### Representation Learning
- Stable dimension (d_s): 32
- Confounding dimension (d_c): 16
- Loss weights: λ_S=1.0, λ_PC=0.5, λ_PF=1.0, λ_DEC=0.1, λ_VAR=0.01
- Learning rate: 1e-3
- Epochs: 100 (with early stopping)

### Outcome Model
- Treatment embedding: 16 dimensions
- Hidden layers: [128, 64]
- Learning rate: 1e-3
- Epochs: 100 (with early stopping)

### Diffusion Model
- Timesteps: 100
- Beta schedule: linear [1e-4, 0.02]
- Hidden layers: [128, 128, 128]
- Learning rate: 1e-3
- Epochs: 100 (with early stopping)

## Data Format

Both CSV files contain visit-level data with columns:

**Required Columns:**
- `RAW_ID`, `UNIQUEID`: Patient identifiers
- `PROTOCOL`: Study protocol
- `VISIT`: Visit number (1, 2, 3, ...)
- `THERAPY_STATUS`: Treatment status
- `THERAPY`: Treatment code
- `THERCODE`: Numeric treatment code
- `AGE`: Patient age
- `ORIGIN`: Geographic origin
- `GENDER`: M/F
- `GEOCODE`: Geographic code
- `HAMD01` - `HAMD17`: Hamilton Depression Rating Scale items

**Preprocessing:**
- Creates next-visit prediction samples: X(t), T(t) → Y(t+1)
- Patient-level train/val/test split (70/15/15)
- Standardizes numeric features
- Encodes categorical features as integers

## Output

For each experiment, the pipeline outputs:

1. **Training Progress**: Loss curves for all three training phases
2. **Test Performance**: MSE and RMSE on held-out test set
3. **Counterfactual Examples**: Sample predictions showing:
   - Observed treatment
   - True outcome
   - Predicted factual outcome
   - Predicted counterfactual outcome
   - Individual treatment effect (ITE)
4. **Treatment Effect Estimates**:
   - Mean and std of ITE across test set
   - Average Treatment Effect (ATE) between common treatments
5. **Saved Models**: Trained models saved to `models_{experiment_name}/`

## Example Output

```
Counterfactual Evaluation
============================================================

Sample Counterfactual Predictions (first 15):
--------------------------------------------------------------------------------
Obs Treatment        True Y       Pred Y (fact)   Pred Y (cf)     ITE
--------------------------------------------------------------------------------
Dulox120            14.237       14.583          13.129          -1.454
VLX150QD            12.458       12.891          13.421          0.530
...

Overall ITE Statistics:
  Mean ITE: -0.3421
  Std ITE:  2.1834

Computing ATE between Dulox120 (baseline) and VLX150QD (target)...
ATE(VLX150QD, Dulox120) = -0.4523
(Positive means VLX150QD leads to higher outcome on average)
```

## CPU-Only Execution

This implementation is designed to run on CPU without requiring GPU. All computations use PyTorch in CPU mode by default.

## Reproducibility

All random seeds are fixed (NumPy, PyTorch, Python random) to ensure reproducible results across runs.

## Citation

If you use this code, please cite the original FCR-CD algorithm paper (see `Treatmenteffect_Simplify_.pdf` for algorithm description).

## License

See repository LICENSE file.
