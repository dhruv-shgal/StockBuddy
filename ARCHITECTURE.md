## Project Architecture and Execution Flow

This document explains how the system fetches data, prepares features, trains models (LSTM/GRU/Transformer), evaluates on a hold‑out period, reconstructs prices from predicted returns, and serves interactive charts in Streamlit.

### High‑Level Flow

1. Inputs (Streamlit sidebar): ticker, date range, model, sequence length, epochs, batch size, future days
2. Data fetch: download adjusted OHLCV from Yahoo Finance via `yfinance`
3. Preprocessing: clean, de‑duplicate, handle infs/NaNs, filter extreme returns (IQR), winsorize tails
4. Feature engineering: RSI, EMA, MACD, day‑of‑week, LogReturn
5. Target creation: next‑step log return of `Close` (stabilizes training)
6. Scaling: MinMaxScaler fit on features and target; sequences created for supervised learning
7. Time‑ordered split: Train → Validation → Test (chronological, no leakage)
8. Model training: chosen model with regularization and callbacks
9. Evaluation: inverse‑transform predicted returns, reconstruct prices, compute metrics, plot
10. Forecast: multi‑step rolling prediction (updates the last window correctly scaled) and future price table

### Key Components and Their Roles

- `fetch_data_yfinance(...)`
  - Pulls adjusted OHLCV for a ticker/date range (auto‑adjusted for splits/dividends)
  - Normalizes column names and ensures required fields exist

- `preprocess_data(df)`
  - Sorts by date; drops duplicates
  - Replaces ±inf with NaN; forward/backward fills small gaps; drops residual NaNs
  - Detects outliers on daily returns using an IQR filter (conservative whiskers)
  - Winsorizes numeric columns at 1st/99th percentiles to cap extreme values

- `engineer_features(df)`
  - Adds technical indicators (RSI14, EMA20, MACD family)
  - Adds `DayOfWeek` and `LogReturn = log(Close).diff()`

- `prepare_data(...)`
  - Defines `feature_cols` (OHLCV + indicators + LogReturn)
  - Sets the target as next‑step `LogReturn` (shifted by −1)
  - Aligns features, base prices, and dates to the target index
  - Scales X and y using MinMaxScaler; builds sliding windows of length `sequence_length`
  - Returns sequences, scalers, dates, and `base_seq` (Close at time t used to reconstruct t+1 price)

- `split_data(X, y, dates, base_seq, train_split, val_split)`
  - Performs chronological split into train/val/test partitions
  - Returns tuples for each split including aligned dates and `base_seq`

- Models
  - LSTM/GRU: stacked recurrent layers with dropout, recurrent_dropout, L2; Dense head
  - Transformer: compact encoder blocks (multi‑head attention + FFN + LayerNorm), pooled then Dense head
  - All compiled with Adam and Huber loss for robustness to outliers

- Training callbacks
  - EarlyStopping (patience), ReduceLROnPlateau, warmup + cosine decay scheduler
  - Mixed precision and XLA (if GPU present) for speed

- `evaluate_and_ensemble(...)`
  - Predicts scaled log returns on the test window for each trained model
  - Inverse‑transforms log returns and reconstructs prices: `price_{t+1} = price_t * exp(logret_pred)`
  - Computes metrics (RMSE, MAE, MAPE, R², Directional Accuracy) against true reconstructed prices
  - Optionally averages models for an equal‑weight ensemble

- Backtesting
  - Simple sign strategy based on predicted changes; computes equity curve and summary metrics (Sharpe, Total Return, Buy & Hold)

- Streamlit UI (`streamlit_app.py`)
  - Sidebar controls for inputs and toggles (e.g., show ensemble on chart)
  - Status messages while fetching → training → evaluating → backtesting
  - Charts: Actual vs Selected Model (red), optional Ensemble; confidence intervals, trading signals
  - Future forecast table: multi‑step rolling prediction of prices using predicted log returns

### Data Leakage Prevention

- Splits are chronological: train, then validation, then test (no shuffling)
- Target is next‑step `LogReturn` computed from aligned feature index (no overlap leakage)

### Scaling and Inverse‑Transform

- MinMaxScaler is fitted on all features and the target
- During evaluation, only the target column is inverse‑transformed; prices are reconstructed using aligned `base_seq`
- During rolling forecast, only the `Close` feature at the last timestep is updated, scaled with the correct column stats

### Configuration Knobs

- `CONFIG` in `app.py`: ticker, dates, sequence length, splits, epochs, batch size, learning rate
- Streamlit sidebar: overrides core settings at runtime without code changes

### Typical Execution Timeline (when you click Train & Predict)

1. Fetch data → preprocess → engineer features → build sequences
2. Chronological split into train/val/test
3. Train the selected model with callbacks and LR schedule
4. Evaluate on test: inverse‑transform returns, reconstruct prices, compute metrics
5. Plot results and run backtest
6. Roll forward to forecast N future days (table with dates/prices)



