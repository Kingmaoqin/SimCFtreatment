# Quick Start Guide

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Verify Installation

```bash
python test_installation.py
```

## 3. Run Experiments

### Option A: Single Seed (Fast)

```bash
# Treatment switched dataset
python main.py --csv_path treatment_switched.csv --experiment_name switched --seed 42

# Treatment consistent dataset
python main.py --csv_path treatment_consistent.csv --experiment_name consistent --seed 42
```

### Option B: Multiple Seeds with Report (Recommended)

```bash
# Run with 3 seeds and generate comprehensive report
python run_experiments_with_report.py \
    --csv_path treatment_switched.csv \
    --experiment_name switched \
    --seeds 42 43 44

python run_experiments_with_report.py \
    --csv_path treatment_consistent.csv \
    --experiment_name consistent \
    --seeds 42 43 44
```

### Option C: Both Datasets (Simple)

```bash
python run_all.py
```

## 4. Output

### Single Seed Run
- Console output with training progress
- Test performance metrics
- Sample counterfactual predictions
- Saved models in `models_{experiment_name}/`

### Multiple Seeds with Report
- Everything from single seed run (for each seed)
- **Comprehensive markdown report** with:
  - Multi-seed averaged results with 95% confidence intervals
  - Individual Treatment Effect (ITE) statistics
  - Average Treatment Effect (ATE) comparisons between all major treatments
  - Statistical significance tests
  - Treatment vs. placebo comparisons (if applicable)

**Report filename:** `report_{experiment_name}_YYYYMMDD_HHMMSS.md`

## 5. Understanding Results

### Key Metrics

- **MSE/RMSE**: Model prediction accuracy (lower is better)
- **ITE (Individual Treatment Effect)**: Expected outcome change for an individual when switching treatments
- **ATE (Average Treatment Effect)**: Population-level average treatment effect

### HAMD Score Interpretation

- **Lower HAMD scores = Better outcomes** (symptom improvement)
- **Negative ATE**: Target treatment leads to lower (better) outcomes
- **Positive ATE**: Target treatment leads to higher (worse) outcomes

### Example Interpretation

```
ATE(VLX150QD, Dulox120) = -0.45
95% CI: [-0.82, -0.08]
```

**Meaning:** VLX150QD leads to 0.45 points lower HAMD scores compared to Dulox120 (baseline), indicating VLX150QD is more effective. The effect is statistically significant since the 95% CI excludes zero.

## 6. Customization

### Adjust Number of Seeds

```bash
python run_experiments_with_report.py \
    --csv_path treatment_switched.csv \
    --experiment_name switched \
    --seeds 42 43 44 45 46  # Use 5 seeds
```

### Change Output Directory

```bash
python run_experiments_with_report.py \
    --csv_path treatment_switched.csv \
    --experiment_name switched \
    --seeds 42 43 44 \
    --output_dir results/
```

## 7. Troubleshooting

### Issue: Data type conversion error

**Solution:** The code has been updated with robust error handling. If you still encounter issues, check that your CSV files have the required columns.

### Issue: Out of memory

**Solution:** Reduce batch size in `data.py` (default is 64, try 32 or 16)

### Issue: Training too slow

**Solution:**
- Reduce number of epochs in training scripts (default is 100)
- Use fewer seeds (e.g., just seed 42)
- Enable GPU if available (automatically detected)

## 8. Expected Runtime

**On CPU:**
- Single seed: ~20-30 minutes per dataset
- 3 seeds: ~60-90 minutes per dataset

**On GPU:**
- Single seed: ~5-10 minutes per dataset
- 3 seeds: ~15-30 minutes per dataset

## 9. Next Steps

1. Review the generated markdown report
2. Examine treatment effect patterns
3. Identify most/least effective treatments
4. Compare results between switched vs. consistent datasets
5. Use insights for clinical decision support

## 10. Citation

If you use this implementation, please cite the FCR-CD algorithm paper (see `Treatmenteffect_Simplify_.pdf`).
