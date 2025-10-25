## Technologies and Tools Used

- Data Acquisition: `yfinance` for historical OHLCV with corporate actions (auto-adjust)
- Feature Engineering: technical indicators (RSI, EMA, MACD), log returns, calendar features
- Modeling: TensorFlow/Keras (LSTM, GRU, compact Transformer encoder)
- Evaluation & Backtesting: NumPy/Pandas; simple long/short strategy and equity curve
- Visualization & UI: Matplotlib for figures, Streamlit for interactive app

## Programming Language

- Python 3.10 (compatible with 3.9–3.12)

## Libraries and Frameworks

- Numerical/Data: `numpy`, `pandas`
- Plotting: `matplotlib`
- Machine Learning / Deep Learning: `tensorflow` (Keras API)
- Data Fetching: `yfinance`
- Preprocessing & Metrics: `scikit-learn` (MinMaxScaler/StandardScaler, metrics)
- App UI: `streamlit`
- (Optional) REST API: `flask`, `flask-cors` for JSON endpoints

## Dataset Source

- Yahoo Finance via `yfinance.download(ticker, start, end, auto_adjust=True)`
- Fields: Open, High, Low, Close, Volume (adjusted for splits/dividends)

## Development Environment

- OS: Windows 10/11
- Python Env: `venv` + `pip` (recommended)
- GPU Acceleration (optional): NVIDIA GPU with CUDA 11.x and cuDNN (Tensor Cores supported); mixed precision enabled
- Editor: Cursor/VS Code (any IDE works)

## Training Setup (Summary)

- Time-ordered splits (Train → Val → Test) to avoid leakage
- Targets: next-step log return; prices reconstructed for plots/metrics
- Scaling: MinMaxScaler for inputs and target (inverse-transform target correctly)
- Regularization: dropout, recurrent_dropout, L2 weight decay, early stopping
- Optimizers/LR: Adam with warmup + cosine decay; ReduceLROnPlateau optional

