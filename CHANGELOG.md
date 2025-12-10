# Changelog

All notable changes to the FCR-CD implementation.

## [2025-12-10] - Latest Updates

### Fixed

#### Data Type Conversion Error ✅
**Issue:** `TypeError: Could not convert string to numeric` during data preprocessing
- **Root Cause:** Mixed-type columns (strings/numbers) in CSV caused conversion failures
- **Solution:**
  - Added `pd.to_numeric()` with `errors='coerce'` for robust type conversion
  - Implemented NaN handling during mean/std computation
  - Added validation for numeric columns before statistical operations
- **Files Modified:** `data.py`

#### Tensor Device Mismatch Error ✅
**Issue:** `RuntimeError: indices should be either on cpu or on the same device as indexed tensor (cpu)`
- **Root Cause:** In diffusion training, `alpha_bars` was on CPU while timestep indices `t` was on GPU
- **Solution:**
  - Move `alpha_bars` to correct device at start of training/validation epochs
  - Smart device-aware indexing in `diffusion_forward_sample()`
  - Falls back to CPU indexing when devices don't match (safe)
  - Uses direct indexing when devices match (fast)
- **Files Modified:** `train_diffusion.py`

### Added

#### Comprehensive Results Reporting System ✨
**Features:**
- Multi-seed experiment support with statistical analysis
- Automated report generation in markdown format
- 95% confidence intervals using t-distribution
- Treatment comparison analysis (all pairs)
- Placebo comparison support
- Statistical significance testing

**New Files:**
- `results_reporter.py` - Report generation module
- `run_experiments_with_report.py` - Multi-seed experiment runner
- `QUICK_START.md` - Detailed usage guide
- `EXAMPLE_REPORT.md` - Sample report output

**Report Includes:**
1. **Model Performance**
   - Test MSE/RMSE with 95% CI
   - Per-seed breakdown

2. **Individual Treatment Effect (ITE)**
   - Mean ITE across seeds with CI
   - ITE distribution statistics

3. **Average Treatment Effect (ATE)**
   - Pairwise treatment comparisons
   - Statistical significance tests
   - Per-seed ATE values

4. **Placebo Analysis** (when applicable)
   - Treatment vs. placebo comparisons
   - Effect size interpretation

**Dependencies Added:**
- `scipy>=1.10.0` for accurate statistical tests

### Enhanced

#### Documentation
- Updated `README.md` with new experiment runners
- Added usage examples for multi-seed experiments
- Included result interpretation guidelines
- Added troubleshooting section

#### Code Robustness
- Better error handling in data loading
- Device-agnostic tensor operations
- Improved type safety

## Usage Examples

### Quick Test (Single Seed)
```bash
python main.py --csv_path treatment_switched.csv --experiment_name switched --seed 42
```

### Recommended: Multi-Seed with Report
```bash
python run_experiments_with_report.py \
    --csv_path treatment_switched.csv \
    --experiment_name switched \
    --seeds 42 43 44
```

### Custom Seeds
```bash
python run_experiments_with_report.py \
    --csv_path treatment_switched.csv \
    --experiment_name switched \
    --seeds 42 43 44 45 46 47  # 6 seeds
```

## Breaking Changes

None. All changes are backward compatible.

## Migration Guide

No migration needed. Existing scripts using `main.py` or `run_all.py` will continue to work.

For enhanced reporting, switch to `run_experiments_with_report.py`:
- Old: `python main.py --csv_path data.csv --experiment_name exp --seed 42`
- New: `python run_experiments_with_report.py --csv_path data.csv --experiment_name exp --seeds 42 43 44`

## Known Issues

None currently.

## Future Enhancements

- [ ] Add support for custom treatment comparisons via config file
- [ ] Export results to CSV/JSON formats
- [ ] Add visualization plots (ITE distributions, ATE comparisons)
- [ ] Support for stratified analysis (by age, gender, etc.)
- [ ] Bayesian credible intervals as alternative to frequentist CI

## Credits

Implementation based on the FCR-CD (Factorized Causal Representation + Confounding Diffusion) algorithm.
