"""
Script to run experiments on both datasets sequentially.

Usage:
    python run_all.py
"""

import sys
from main import run_experiment


def main():
    """Run experiments on both datasets."""

    print("\n" + "#"*80)
    print("# Running FCR-CD Algorithm on Both Datasets")
    print("#"*80 + "\n")

    seed = 42

    # Experiment 1: Treatment Switched
    print("\n" + "#"*80)
    print("# EXPERIMENT 1: Treatment Switched Dataset")
    print("#"*80 + "\n")

    try:
        run_experiment(
            csv_path='treatment_switched.csv',
            experiment_name='switched',
            seed=seed
        )
    except Exception as e:
        print(f"\n❌ Error in switched experiment: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 2: Treatment Consistent
    print("\n" + "#"*80)
    print("# EXPERIMENT 2: Treatment Consistent Dataset")
    print("#"*80 + "\n")

    try:
        run_experiment(
            csv_path='treatment_consistent.csv',
            experiment_name='consistent',
            seed=seed
        )
    except Exception as e:
        print(f"\n❌ Error in consistent experiment: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#"*80)
    print("# ALL EXPERIMENTS COMPLETED")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
