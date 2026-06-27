# Central Bank Speech Alpha — EUR/USD Event Study

Fed and ECB speech sentiment → EUR/USD directional signal using **CentralBankRoBERTa-sentiment-classifier**, validated through permutation placebo tests and walk-forward analysis.

**Champion strategy**: `W=13 step_4tier` — statistically significant at **p=0.0006** (99.94% confidence), robust out-of-sample.

```bash
python src/event_study.py    # Full optimization report
```

## Approach

1. **Data**: `istat-ai/ECB-FED-speeches` (HuggingFace) + yfinance EUR/USD daily bars (10y)
2. **Sentiment**: `CentralBankRoBERTa-sentiment-classifier` — 88% sentiment accuracy, 93% agent classifier
3. **Scoring**: Agent whitelist `{'Central Bank'}`, keyword ≥2, strict_score ≥3, mean daily aggregation
4. **Signal**: Country-split (Fed + ECB scored independently), non-filled aggregation
5. **Validation**: Permutation placebo (5,000–10,000 shuffles), expanding walk-forward OOS

## Results

### Champion: W=13 step_4tier

| Metric | Value |
|--------|-------|
| Window | 13 trading days (~3 weeks) |
| Events | 109 active (182 total, 73 zeroed by floor) |
| Hit rate | 57.8% |
| Avg return/event | +0.29% |
| Total net return | +30.8% (10 years) |
| Sharpe (ann.) | 1.373 |
| **p-value** | **0.0006 ***** |

### Position function (`step_4tier`)

| Score range | Position |
|------------|----------|
| abs < 0.2 | 0 (zero out noise) |
| 0.2 – 0.4 | ±0.33 |
| 0.4 – 0.6 | ±0.66 |
| ≥ 0.6 | ±1.00 |

### Walk-forward OOS

| OOS Period | Events | Hit Rate | p-value |
|-----------|--------|----------|---------|
| Y6–7 (2yr) | 22 | 54.5% | 0.0050 *** |
| Y6–8 (3yr) | 40 | 50.0% | 0.0115 ** |
| Y6–10 (5yr) | 56 | 58.9% | 0.0050 *** |

All OOS windows statistically significant at p<0.05.

### Asymmetry

- **Dovish** (129 events, 71%): Drives the signal — W=1 Sharpe 2.179, score/1d-ret corr +0.089
- **Hawkish** (53 events, 29%): Fails at short windows, recovers at W=13
- Position function handles asymmetry via lower effective weights on hawkish events with abs < 0.4

### Evolution

| Iteration | Approach | Best p-value |
|-----------|----------|-------------|
| Baseline | H4 bars, EWM pooled, ModernFinBERT | 0.488 |
| V1 | Daily bars, non-filled country-split, CentralBankRoBERTa | 0.111 |
| V2 | Stricter agent/keyword filter (Central Bank only, ≥2 kw, ≥3 strict) | 0.015 |
| V3 | Fed-only event study, binary position | 0.028 |
| V4 | Continuous linear sizing | 0.003 |
| **V5** | **step_4tier + floor at 0.2** | **0.0006** |

## Project Structure

```
├── src/
│   ├── event_study.py            # Optimization engine + final report
│   ├── sentiment_pipeline.py     # CentralBankRoBERTa scoring
│   ├── align_and_merge.py        # Daily alignment + feature engineering
│   ├── backtest_engine.py        # Daily regression backtests
│   ├── causality_analysis.py     # Granger causality + feature importance
│   ├── placebo_test.py           # Permutation tests for daily model
│   ├── fetch_data.py             # FX + speech ingestion
│   └── fred_controls.py          # FRED macro shocks
├── data/                          # merged_daily.csv, speeches_scored.csv
├── main.py                        # Full pipeline orchestrator
├── requirements.txt
└── environment.yml
```

## Reproducibility

All scripts use `seed=42` with `enforce_strict_reproducibility()` across random, numpy, and torch. Placebo tests shuffle speech scores (not returns) to preserve market microstructure while destroying the speech-return relationship.
