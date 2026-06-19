import os
import random
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


def generate_correlation_dashboard():
    """Per-pair correlation heatmap and lag shape analysis across the multi-currency panel."""
    input_path = os.path.join('data', 'currency_panel_h4.csv')
    output_dir = 'notebooks'
    output_path = os.path.join(output_dir, 'correlation_shapes.png')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing panel data at {input_path}")

    df = pd.read_csv(input_path)

    lag_cols = [f'speech_lag_{i}' for i in range(1, 7)]
    pairs = df['pair_id'].unique()
    n_pairs = len(pairs)

    fig, axes = plt.subplots(2, n_pairs, figsize=(7 * n_pairs, 14))

    if n_pairs == 1:
        axes = axes.reshape(2, 1)

    for j, pid in enumerate(pairs):
        sub = df[df['pair_id'] == pid]
        core_features = ['returns', 'econ_surprise', 'returns_lag1'] + lag_cols
        corr_matrix = sub[core_features].corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".3f",
            cmap='coolwarm', vmin=-0.1, vmax=0.1, center=0,
            square=True, linewidths=.5,
            cbar_kws={"shrink": .8, "label": "Pearson r"},
            ax=axes[0, j]
        )
        axes[0, j].set_title(f"{pid} — Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=15)
        label_mappings = ['Returns', 'FRED Shock', 'Returns (L1)'] + [f'L{i}' for i in range(1, 7)]
        axes[0, j].set_xticklabels(label_mappings, rotation=45, ha='right')
        axes[0, j].set_yticklabels(label_mappings, rotation=0)

        lag_correlations = [corr_matrix.loc['returns', col] for col in lag_cols]
        lag_labels = [f'{i * 4}h' for i in range(1, 7)]

        axes[1, j].plot(lag_labels, lag_correlations, marker='o', linewidth=2.5, color='tab:blue', label=f'{pid} Shape')
        axes[1, j].axhline(0, color='black', linestyle=':', alpha=0.6)

        peak_idx = np.argmax(np.abs(lag_correlations))
        axes[1, j].scatter(lag_labels[peak_idx], lag_correlations[peak_idx],
                           color='tab:red', s=150, zorder=5, label=f'Peak at {lag_labels[peak_idx]}')

        axes[1, j].set_title(f"{pid} — Signal Digestion Curve", fontsize=13, fontweight='bold', pad=15)
        axes[1, j].set_xlabel("Time Since Speech", fontsize=11, labelpad=10)
        axes[1, j].set_ylabel("Correlation with Returns", fontsize=11, labelpad=10)
        axes[1, j].grid(True, linestyle='--', alpha=0.3)
        axes[1, j].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Panel correlation dashboard output to: {output_path}")


if __name__ == "__main__":
    generate_correlation_dashboard()
