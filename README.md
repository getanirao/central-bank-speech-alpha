# FX Quant Language Model Pipeline

**Central Bank Speech Sentiment → EUR/USD Return Prediction**

A quantitative finance pipeline that uses HuggingFace's `FinancialBERT-Sentiment-Analysis` to extract hawkish/dovish semantic scores from ECB and Fed speeches, aligns them with EUR/USD price action on 4-hour bars, controls for real FRED macroeconomic shocks (CPI, NFP), and statistically validates the predictive edge via distributed-lag OLS regression and walk-forward out-of-sample backtesting.

## Architecture

```
speeches ──> FinancialBERT ──> semantic_score ──┐
                                                 ├──> distributed lags (lag_1..lag_6) ──> OLS ──> signal
EUR/USD ──> yfinance H4 bars ──> returns ───────┤
                                                 │
FRED CPI/NFP ──> econ_surprise Z-score ─────────┘
```

## Results

### Multi-Lag OLS (Distributed Lag Model, n=3,176 4h bars)

| Lag | Window | Coefficient | p-value | Significant |
|---|---|---|---|---|
| `speech_lag_1` | 0–4h | -0.0004 | 0.461 | |
| `speech_lag_2` | 4–8h | -3.9e-05 | 0.959 | |
| `speech_lag_3` | 8–12h | -0.0006 | 0.421 | |
| **`speech_lag_4`** | **12–16h** | **+0.0017** | **0.026** | **✅** |
| `speech_lag_5` | 16–20h | -0.0005 | 0.540 | |
| `speech_lag_6` | 20–24h | -0.0001 | 0.827 | |
| `econ_surprise` | — | **+0.0003** | **0.000** | **✅** |

Model fit: **R² = 0.022**, F-test p = 2.86e-12

### Granger Causality (semantic_regime → returns)

| Lags | p-value | Significant |
|---|---|---|
| 1 (4h) | 0.0490 | ✅ |
| 2 (8h) | 0.1103 | |
| 3 (12h) | 0.1506 | |
| 4 (16h) | 0.0486 | ✅ |
| 5 (20h) | 0.0591 | |
| 6 (24h) | 0.1115 | |

### Out-of-Sample Walk-Forward Backtest (70/30 chronological split)

| Metric | Value |
|---|---|
| **OOS R²** | **+0.00434** ✅ |
| **Directional Hit Rate** | **60.97%** |
| **Information Ratio (annual.)** | **0.4837** |
| **OOS Strategy Return** | **+33.25%** |
| **OOS Market Return** | **+20.51%** |

### Key Insight: 16-Hour Institutional Rebalancing Effect

Speech semantics carry **zero predictive power in the first 12 hours** (algorithmic noise and headline scalping dominate). At **16 hours**, the signal becomes statistically significant — the exact window where institutional asset allocators complete portfolio digestion and execute block orders.

## Project Structure

```
├── src/
│   ├── fetch_data.py             # FX + speech data ingestion
│   ├── sentiment_pipeline.py     # FinancialBERT + topic filter
│   ├── fred_controls.py          # FRED CPI/NFP macro shocks
│   ├── align_and_merge.py        # Distributed lags + merge
│   ├── causality_analysis.py     # Multi-lag OLS + Granger + 3-panel plot
│   ├── backtest_engine.py        # Walk-forward OOS validation
│   └── live_pipeline.py          # Real-time signal engine
├── notebooks/
│   └── shape_analysis.png        # 3-panel diagnostic chart
├── main.py                       # Full orchestrator
├── requirements.txt
└── .gitignore
```

## Usage

```bash
# Full pipeline (download data, train, test)
python main.py

# Live signal for current 4h candle
python src/live_pipeline.py
```

## Dependencies

`transformers`, `torch`, `yfinance`, `pandas`, `numpy`, `statsmodels`, `datasets`, `scikit-learn`, `matplotlib`, `fredapi`

## Data Sources

- **Central bank speeches**: `istat-ai/ECB-FED-speeches` (HuggingFace Datasets)
- **FX prices**: Yahoo Finance (`EURUSD=X`, 1h bars)
- **Macroeconomic controls**: FRED API (`CPIAUCNS`, `PAYEMS`)

## Methodology

1. **Topic Filter**: Speeches must contain ≥3 policy keywords (inflation, interest rate, hawkish, etc.)
2. **Sentiment Scoring**: FinancialBERT maps speech text → numeric score (+=hawkish, -=dovish)
3. **Distributed Lags**: 6 sequential 4-hour lag columns capture the delayed price digestion curve
4. **OLS Regression**: `returns ~ lag_1 + lag_2 + ... + lag_6 + econ_surprise + returns_lag1`
5. **Walk-Forward**: 70% historical train → predict next 30% out-of-sample, metrics computed on unseen data
