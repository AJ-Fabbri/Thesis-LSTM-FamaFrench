# Neural Networks in Asset Management: LSTM Price Forecasting

**Ray O. Werner Thesis Prize Winner** — Colorado College, May 2025

This award is given to "the senior thesis judged the most outstanding" across the Economics & Business Department at Colorado College.

> “We felt that AJ Fabbri examined a novel and timely question as markets and managers begin to rely more on artificial intelligence. [...] AJ also employed methodological and computational techniques that were rigorous, appropriate, and impressive." —Ray O. Werner Prize Committee

This notebook implements the MLS-LSTM (Multi-Layered Sequential Long Short-Term Memory) model from my paper, which tested whether augmenting LSTM networks with Fama-French factor data improves alpha generation across market segments.

I remember the entire process of this project very fondly, and this code is where I first learned how to write a data pipeline and build a neural net.

Feel free to download the presentation from this repo or read the full paper at [https://digitalcc.coloradocollege.edu/record/8037?ln=en&p=AJ+Fabbri&v=pdf](https://digitalcc.coloradocollege.edu/record/8037?ln=en&p=AJ+Fabbri&v=pdf)

## What it does

1. **Data ingestion:** Downloads up to 20 years of daily OHLCV data for any ticker via `yfinance` and merges it with Fama-French five-factor data (Mkt-RF, SMB, HML, RMW, CMA, RF)
2. **Preprocessing:** Scales features with `MinMaxScaler` and builds sliding windows (15-day input -> 5-day forecast)
3. **Model training:** Tunes a stacked LSTM architecture using Bayesian hyperparameter optimization via `keras-tuner` (32–512 units per layer, 1–4 hidden LSTM layers, dropout, dense activation)
4. **Evaluation:** Reports MSE, MAE, and R² on held-out test data
5. **Trading simulation:** Generates volatility-adjusted buy/sell signals from multi-step price forecasts and backtests an active strategy against a buy-and-hold benchmark, calculating alpha (defined as outperformance compared to buy and hold).

## Key finding

Incorporating Fama-French factors boosted alpha to a statistically significant degree across 30 tested assets (large-cap stocks, small-caps, and cryptocurrencies), with the largest gains on low-information assets like crypto consistent with greater market inefficiency in those segments.

Limitations included sample size and one market regime at the time of the study. If I were to develop this further, I'd automate the pipeline and add full cross-validation to test across multiple market regimes. 

## Requirements

```
tensorflow / keras
keras-tuner
yfinance
pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels
```

Set `ticker` and `maindir` at the top of the notebook, then run cells top to bottom. Fama-French data (`FF_Factors.csv`) must be placed in `maindir/data/`.

<img width="1485" height="1536" alt="1746663196792-2" src="https://github.com/user-attachments/assets/6d77bbcf-d4a8-4a99-854f-cbb0f625e851" />
