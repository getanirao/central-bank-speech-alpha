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
    input_path = os.path.join('data', 'merged_daily.csv')
    output_dir = 'notebooks'
    output_path = os.path.join(output_dir, 'correlation_shapes.png')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing data at {input_path}")

    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    lag_cols = [f'{p}_lag_{i}' for p in ['fed', 'ecb'] for i in range(1, 7)]
    core_features = ['returns', 'econ_surprise', 'returns_lag1'] + lag_cols
    corr_matrix = df[core_features].corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f",
                cmap='coolwarm', vmin=-0.1, vmax=0.1, center=0,
                square=True, linewidths=.5,
                cbar_kws={"shrink": .8, "label": "Pearson r"}, ax=ax1)
    ax1.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
    lbls = ['Returns', 'FRED Shock', 'Returns(L1)']
    lbls += [f'F{i}' for i in range(1, 7)] + [f'E{i}' for i in range(1, 7)]
    ax1.set_xticklabels(lbls, rotation=45, ha='right')
    ax1.set_yticklabels(lbls, rotation=0)

    # Two digestion curves: Fed and ECB
    fed_corr = [corr_matrix.loc['returns', f'fed_lag_{i}'] for i in range(1, 7)]
    ecb_corr = [corr_matrix.loc['returns', f'ecb_lag_{i}'] for i in range(1, 7)]
    lag_labels = [f'{i}D' for i in range(1, 7)]

    ax2.plot(lag_labels, fed_corr, marker='o', linewidth=2.5, color='tab:orange', label='Fed')
    ax2.plot(lag_labels, ecb_corr, marker='s', linewidth=2.5, color='tab:green', label='ECB')
    ax2.axhline(0, color='black', linestyle=':', alpha=0.6)
    ax2.set_title("Signal Digestion Curve by Country", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Time Since Speech", fontsize=11, labelpad=10)
    ax2.set_ylabel("Correlation with Returns", fontsize=11, labelpad=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Dashboard saved to: {output_path}")


if __name__ == "__main__":
    generate_correlation_dashboard()
