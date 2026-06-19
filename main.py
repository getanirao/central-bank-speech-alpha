import os
import sys
import random
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()

from fetch_data import create_directory_structure, fetch_fx_data, fetch_speech_corpus
from fred_controls import fetch_and_save_fred_shocks
from sentiment_pipeline import run_sentiment_analysis
from align_and_merge import align_and_merge_datasets
from causality_analysis import execute_statistical_tests
from backtest_engine import run_comparison_backtest
from visualize_correlation import generate_correlation_dashboard
from placebo_test import run_placebo_test


def run_pipeline(training_samples=500):
    print("=" * 70)
    print("     FX QUANT LANGUAGE MODEL PIPELINE")
    print("=" * 70)

    model_dir = os.path.join('models', 'modernfinbert_finetuned')
    model_ready = os.path.exists(model_dir) and any(
        f.endswith('.safetensors') for f in os.listdir(model_dir))
    if not model_ready:
        print("\n--- Phase 0: Fine-Tuning ModernFinBERT ---")
        from train_sentiment import train_model
        train_model(max_samples=training_samples)
    else:
        print("\nFine-tuned ModernFinBERT found.")

    create_directory_structure()
    fetch_fx_data(ticker="EURUSD=X", period="2y", interval="1h")
    fetch_speech_corpus()

    print("\n--- Phase 1: FRED Macro ---")
    fetch_and_save_fred_shocks()

    print("\n--- Phase 2: ModernFinBERT Sentiment ---")
    run_sentiment_analysis(batch_size=16)

    print("\n--- Phase 3: Merge & Align ---")
    align_and_merge_datasets()

    print("\n--- Phase 4: Econometric Tests ---")
    execute_statistical_tests()

    print("\n--- Phase 5: Backtest ---")
    run_comparison_backtest(train_window_pct=0.70)

    print("\n--- Phase 6: Placebo Test (1000x) ---")
    run_placebo_test(n_iterations=1000)

    print("\n--- Phase 7: Correlation Diagnostics ---")
    generate_correlation_dashboard()


if __name__ == "__main__":
    run_pipeline()
