# FX Quant Language Model Pipeline

Central bank speech sentiment → EUR/USD H4 return prediction using ModernFinBERT, distributed-lag regression, stacking ensemble, and purged cross-validation.

```bash
python main.py          # Full pipeline
python src/live_pipeline.py  # Live signal
```

## Pipeline

1. **Data**: yfinance EUR/USD 1h bars, istat-ai/ECB-FED-speeches (HuggingFace), FRED CPI/NFP
2. **Sentiment**: ModernFinBERT → `bearish→+score, bullish→-score, neutral→0`
3. **Align**: 12 distributed lags (6 broad + 6 strict), ewm decay (24h half-life), Almon PDL, engineered features (hv_20, interaction, momentum)
4. **Models**: OLS, Ridge, XGBoost (max_depth=2, L1/L2), Almon PDL, Stacking Ensemble (Ridge+XGBoost→ElasticNet), Regime-Adaptive (vol tercile)
5. **Validation**: PurgedKFold (5-fold, 4h purge/embargo), rolling walk-forward (6m/2m), placebo permutation (1000x)

## Results (Purged CV)

| Model | OOS R² | Hit Rate |
|---|---|---|
| Stacking Ensemble | -0.00063 | 47.9% |
| Ridge (Purged CV) | -0.00771 | 47.8% |
| Rolling Ridge (avg) | -0.00830 | 46.8% |
| Placebo p-value | 0.488 | — |

## Project Structure

```
├── src/
│   ├── fetch_data.py             # FX + speech ingestion
│   ├── train_sentiment.py        # Fine-tune on FOMC hawkish/dovish
│   ├── sentiment_pipeline.py     # ModernFinBERT scoring
│   ├── fred_controls.py          # FRED macro shocks
│   ├── align_and_merge.py        # Distributed lags + features
│   ├── causality_analysis.py     # OLS/Ridge/XGBoost + Granger
│   ├── backtest_engine.py        # All backtests + purged CV
│   ├── placebo_test.py           # Permutation test
│   ├── live_pipeline.py          # Real-time signal
│   └── visualize_correlation.py  # Correlation diagnostics
├── main.py                       # Orchestrator
├── requirements.txt
└── environment.yml
```
