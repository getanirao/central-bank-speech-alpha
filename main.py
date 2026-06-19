import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

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

    # --- Phase -1: Fine-tune ModernFinBERT (if not already done) ---
    model_dir = os.path.join('models', 'modernfinbert_finetuned')
    model_ready = os.path.exists(model_dir) and any(
        f.endswith('.safetensors') for f in os.listdir(model_dir))
    if not model_ready:
        print("\n--- Phase 0: Fine-Tuning ModernFinBERT on Financial Sentiment ---")
        from train_sentiment import train_model
        train_model(max_samples=training_samples)
    else:
        print("\nFine-tuned ModernFinBERT found — skipping training.")

    create_directory_structure()
    fetch_fx_data(ticker="EURUSD=X", period="2y", interval="1h")
    fetch_speech_corpus()

    print("\n--- Phase 1: Executing FRED Macro Engine Realignment ---")
    fetch_and_save_fred_shocks()

    print("\n--- Phase 2: Processing Filtered Text through ModernFinBERT ---")
    run_sentiment_analysis(batch_size=16)

    print("\n--- Phase 3: Merging Continuous Time-Series Horizons ---")
    align_and_merge_datasets()

    print("\n--- Phase 4: Executing Multi-Lag Distributed Econometric Testing ---")
    execute_statistical_tests()

    print("\n--- Phase 5: Executing Forward Walk Out-of-Sample Backtest ---")
    run_comparison_backtest(train_window_pct=0.70)

    print("\n--- Phase 6: Permutation / Placebo Test (1000x shuffle) ---")
    run_placebo_test(n_iterations=1000)

    print("\n--- Phase 7: Generating Diagnostic Correlation Heatmaps & Shapes ---")
    generate_correlation_dashboard()


if __name__ == "__main__":
    run_pipeline()
