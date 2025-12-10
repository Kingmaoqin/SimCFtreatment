"""
Results reporting module for treatment effect estimation.
Generates comprehensive markdown reports with statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os


class ResultsReporter:
    """Generates comprehensive treatment effect evaluation reports."""

    def __init__(self, experiment_name: str, output_dir: str = "."):
        """
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save reports
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.results = []
        self.ate_results = []

    def add_seed_result(self, seed: int, metrics: Dict):
        """
        Add results from a single seed run.

        Args:
            seed: Random seed used
            metrics: Dictionary with performance metrics
        """
        result = {'seed': seed}
        result.update(metrics)
        self.results.append(result)

    def add_ate_comparison(self, seed: int, baseline: str, target: str,
                          ate: float, n_samples: int):
        """
        Add ATE comparison between treatments.

        Args:
            seed: Random seed
            baseline: Baseline treatment name
            target: Target treatment name
            ate: Average treatment effect
            n_samples: Number of samples used
        """
        self.ate_results.append({
            'seed': seed,
            'baseline': baseline,
            'target': target,
            'ate': ate,
            'n_samples': n_samples
        })

    def compute_confidence_interval(self, values: List[float],
                                   confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute mean and confidence interval.

        Args:
            values: List of values
            confidence: Confidence level (default 0.95)

        Returns:
            mean, lower_ci, upper_ci
        """
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        n = len(values)

        # Compute margin of error (using t-distribution approximation)
        if n > 1:
            try:
                from scipy import stats
                t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
            except ImportError:
                # Fallback to z-score if scipy not available
                t_critical = 1.96  # Approximate 95% CI
            margin = t_critical * (std / np.sqrt(n))
        else:
            margin = 0.0

        lower_ci = mean - margin
        upper_ci = mean + margin

        return mean, lower_ci, upper_ci

    def generate_report(self) -> str:
        """
        Generate comprehensive markdown report.

        Returns:
            Markdown formatted report string
        """
        lines = []

        # Header
        lines.append(f"# Treatment Effect Evaluation Report: {self.experiment_name}")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # 1. Overview
        lines.append("## 1. Experiment Overview\n")
        lines.append(f"- **Experiment Name:** {self.experiment_name}")
        lines.append(f"- **Number of Seeds:** {len(self.results)}")
        if self.results:
            lines.append(f"- **Seeds Used:** {', '.join(str(r['seed']) for r in self.results)}")
        lines.append("\n---\n")

        # 2. Multi-Seed Performance Summary
        if self.results:
            lines.append("## 2. Model Performance Across Seeds\n")
            lines.append("### 2.1 Test Set Performance\n")

            # Collect metrics
            test_mse = [r.get('test_mse', np.nan) for r in self.results]
            test_rmse = [r.get('test_rmse', np.nan) for r in self.results]

            # Compute statistics
            mse_mean, mse_lower, mse_upper = self.compute_confidence_interval(test_mse)
            rmse_mean, rmse_lower, rmse_upper = self.compute_confidence_interval(test_rmse)

            # Table
            lines.append("| Metric | Mean | 95% CI | Std Dev |")
            lines.append("|--------|------|--------|---------|")
            lines.append(f"| **MSE** | {mse_mean:.4f} | [{mse_lower:.4f}, {mse_upper:.4f}] | {np.std(test_mse):.4f} |")
            lines.append(f"| **RMSE** | {rmse_mean:.4f} | [{rmse_lower:.4f}, {rmse_upper:.4f}] | {np.std(test_rmse):.4f} |")
            lines.append("")

            # Per-seed details
            lines.append("### 2.2 Per-Seed Results\n")
            lines.append("| Seed | Test MSE | Test RMSE |")
            lines.append("|------|----------|-----------|")
            for r in self.results:
                lines.append(f"| {r['seed']} | {r.get('test_mse', np.nan):.4f} | {r.get('test_rmse', np.nan):.4f} |")
            lines.append("\n---\n")

        # 3. Individual Treatment Effect (ITE) Statistics
        if self.results and any('ite_mean' in r for r in self.results):
            lines.append("## 3. Individual Treatment Effect (ITE) Analysis\n")

            ite_means = [r.get('ite_mean', np.nan) for r in self.results if 'ite_mean' in r]
            ite_stds = [r.get('ite_std', np.nan) for r in self.results if 'ite_std' in r]

            if ite_means:
                mean_ite, ite_lower, ite_upper = self.compute_confidence_interval(ite_means)
                mean_std = np.mean(ite_stds)

                lines.append("### 3.1 ITE Summary Across Seeds\n")
                lines.append("| Statistic | Value | 95% CI |")
                lines.append("|-----------|-------|--------|")
                lines.append(f"| **Mean ITE** | {mean_ite:.4f} | [{ite_lower:.4f}, {ite_upper:.4f}] |")
                lines.append(f"| **Average Std Dev** | {mean_std:.4f} | - |")
                lines.append("")

                lines.append("### 3.2 ITE Distribution by Seed\n")
                lines.append("| Seed | Mean ITE | Std Dev ITE |")
                lines.append("|------|----------|-------------|")
                for r in self.results:
                    if 'ite_mean' in r:
                        lines.append(f"| {r['seed']} | {r['ite_mean']:.4f} | {r['ite_std']:.4f} |")
                lines.append("\n---\n")

        # 4. Average Treatment Effect (ATE) Comparisons
        if self.ate_results:
            lines.append("## 4. Average Treatment Effect (ATE) Analysis\n")

            # Group by comparison pairs
            ate_df = pd.DataFrame(self.ate_results)
            comparisons = ate_df.groupby(['baseline', 'target'])

            lines.append("### 4.1 Treatment Comparisons\n")

            for (baseline, target), group in comparisons:
                lines.append(f"#### {target} vs. {baseline} (Baseline)\n")

                ate_values = group['ate'].values
                n_samples = group['n_samples'].values[0] if len(group) > 0 else 0

                ate_mean, ate_lower, ate_upper = self.compute_confidence_interval(ate_values)

                lines.append(f"- **Number of Test Samples:** {n_samples}")
                lines.append(f"- **Mean ATE:** {ate_mean:.4f}")
                lines.append(f"- **95% CI:** [{ate_lower:.4f}, {ate_upper:.4f}]")
                lines.append(f"- **Std Dev:** {np.std(ate_values):.4f}")

                # Interpretation
                if ate_mean > 0:
                    lines.append(f"- **Interpretation:** {target} leads to **higher** outcomes compared to {baseline} (positive ATE)")
                elif ate_mean < 0:
                    lines.append(f"- **Interpretation:** {target} leads to **lower** outcomes compared to {baseline} (negative ATE)")
                else:
                    lines.append(f"- **Interpretation:** No difference between {target} and {baseline}")

                # Check if CI includes zero
                if ate_lower <= 0 <= ate_upper:
                    lines.append("- **Statistical Significance:** *Not significant* (95% CI includes zero)")
                else:
                    lines.append("- **Statistical Significance:** *Significant* (95% CI excludes zero)")

                lines.append("")

                # Per-seed details
                lines.append("**Per-Seed ATE Values:**\n")
                lines.append("| Seed | ATE |")
                lines.append("|------|-----|")
                for _, row in group.iterrows():
                    lines.append(f"| {row['seed']} | {row['ate']:.4f} |")
                lines.append("")

            lines.append("---\n")

        # 5. Placebo Comparisons (if available)
        placebo_results = [r for r in self.ate_results if 'placebo' in r['baseline'].lower() or 'placebo' in r['target'].lower()]
        if placebo_results:
            lines.append("## 5. Treatment vs. Placebo Analysis\n")

            placebo_df = pd.DataFrame(placebo_results)
            for (baseline, target), group in placebo_df.groupby(['baseline', 'target']):
                # Determine which is placebo
                is_baseline_placebo = 'placebo' in baseline.lower()
                active_treatment = target if is_baseline_placebo else baseline
                placebo_name = baseline if is_baseline_placebo else target

                lines.append(f"### {active_treatment} vs. {placebo_name}\n")

                ate_values = group['ate'].values
                if not is_baseline_placebo:
                    ate_values = -ate_values  # Flip sign if placebo is target

                ate_mean, ate_lower, ate_upper = self.compute_confidence_interval(ate_values)

                lines.append(f"- **Mean ATE:** {ate_mean:.4f}")
                lines.append(f"- **95% CI:** [{ate_lower:.4f}, {ate_upper:.4f}]")

                if ate_mean < 0:
                    lines.append(f"- **Interpretation:** {active_treatment} shows **benefit** over placebo (negative ATE means lower HAMD scores)")
                elif ate_mean > 0:
                    lines.append(f"- **Interpretation:** {active_treatment} shows **worse** outcomes than placebo")
                else:
                    lines.append(f"- **Interpretation:** No difference from placebo")

                lines.append("")

            lines.append("---\n")

        # 6. Summary and Conclusions
        lines.append("## 6. Summary\n")
        if self.results:
            lines.append(f"✓ Evaluated {len(self.results)} independent seed runs")
        if self.ate_results:
            unique_comparisons = len(set((r['baseline'], r['target']) for r in self.ate_results))
            lines.append(f"✓ Analyzed {unique_comparisons} treatment comparison(s)")
        lines.append(f"✓ All results include 95% confidence intervals")
        lines.append("\n---\n")

        # Footer
        lines.append("## Notes\n")
        lines.append("- **Outcome:** HAMD (Hamilton Depression Rating Scale) total score at next visit")
        lines.append("- **Lower HAMD scores indicate better outcomes (symptom improvement)**")
        lines.append("- **Negative ATE:** Target treatment leads to lower (better) outcomes")
        lines.append("- **Positive ATE:** Target treatment leads to higher (worse) outcomes")
        lines.append("- **CI:** Confidence Interval (95% unless otherwise specified)")

        return "\n".join(lines)

    def save_report(self, filename: Optional[str] = None):
        """
        Save report to markdown file.

        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"report_{self.experiment_name}_{timestamp}.md"

        filepath = os.path.join(self.output_dir, filename)

        report = self.generate_report()

        with open(filepath, 'w') as f:
            f.write(report)

        print(f"\n✓ Report saved to: {filepath}")

        return filepath
