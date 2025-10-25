"""
Stock price prediction pipeline

Self-contained script that:
1) Fetches market data via yfinance
2) Engineers technical indicators and scales features
3) Creates sequences for supervised learning
4) Trains LSTM, GRU, and a light Transformer model
5) Evaluates, ensembles, and backtests
"""

import os
import math
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("yfinance is required. Install with: pip install yfinance")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras import mixed_precision

# API server
from flask import Flask, request, jsonify
from flask_cors import CORS

# --------------------------
# Configuration
# --------------------------
CONFIG = {
    'ticker': 'AAPL',
    'start_date': '2020-01-01',
    'end_date': datetime.date.today().isoformat(),
    'sequence_length': 90,
    'train_split': 0.7,
    'val_split': 0.15,
    'epochs': 20,
    'batch_size': 16,
    'initial_capital': 10000.0,
    'learning_rate': 1e-4,
    'seed': 42
}

np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

# --------------------------
# GPU/Performance configuration
# --------------------------
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        # Enable mixed precision for tensor cores (speed-up on modern GPUs)
        mixed_precision.set_global_policy('mixed_float16')
        # Enable XLA compilation (can improve performance)
        tf.config.optimizer.set_jit(True)
        print(f"Using GPU(s): {[d.name for d in gpus]}")
    else:
        print("No GPU detected; training will use CPU.")
except Exception as e:
    print(f"GPU configuration warning: {e}")


# --------------------------
# Utilities
# --------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    # Ensure index alignment to avoid scalar-construction errors
    return pd.DataFrame(
        {
            'MACD': np.asarray(macd).ravel(),
            'MACD_signal': np.asarray(signal_line).ravel(),
            'MACD_hist': np.asarray(hist).ravel()
        },
        index=series.index
    )


def create_sequences(data_array: np.ndarray, target_array: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data_array)):
        X.append(data_array[i - seq_len:i])
        y.append(target_array[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Directional accuracy
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    da = (true_dir == pred_dir).mean() * 100.0
    # MAPE
    eps = 1e-9
    mape = (np.abs((y_true - y_pred) / (y_true + eps))).mean() * 100.0
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + eps
    r2 = 1 - (ss_res / ss_tot)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Directional_Accuracy': da}


def set_plot_style():
    plt.style.use('seaborn-v0_8')


def plot_predictions(dates, y_true, pred_dict, title="Predictions"):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label='Actual', color='black', linewidth=2)
    for name, preds in pred_dict.items():
        plt.plot(dates, preds, label=name)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confidence_intervals(dates, y_true, y_pred, ci_bounds):
    lower, upper = ci_bounds
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label='Actual', color='black')
    plt.plot(dates, y_pred, label='Ensemble', color='green')
    plt.fill_between(dates, lower, upper, color='green', alpha=0.15, label='Confidence band')
    plt.legend(); plt.tight_layout(); plt.show()


def create_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics_dict).T.sort_values('RMSE')


# --------------------------
# Data pipeline
# --------------------------
def inverse_target_only(pred_scaled: np.ndarray, scaler, num_features: int, target_index: int) -> np.ndarray:
    """Inverse-transform a 1D scaled target using a scaler fitted on multiple features.
    Reconstructs a dummy matrix with only the target column populated, then picks back that column.
    """
    mat = np.zeros((len(pred_scaled), num_features), dtype=np.float32)
    mat[:, target_index] = pred_scaled
    inv = scaler.inverse_transform(mat)
    return inv[:, target_index]


def scale_single_feature(value: float, scaler, feature_index: int) -> float:
    """Scale a single raw feature value using a fitted StandardScaler or MinMaxScaler that was fit on all features."""
    # StandardScaler case
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        return float((value - scaler.mean_[feature_index]) / (scaler.scale_[feature_index] + 1e-12))
    # MinMaxScaler case
    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'scale_'):
        return float((value - scaler.data_min_[feature_index]) * scaler.scale_[feature_index])
    # Fallback: reconstruct a row
    row = np.zeros((1, len(getattr(scaler, 'scale_', [0]* (feature_index+1)))), dtype=np.float32)
    row[0, feature_index] = value
    return float(scaler.transform(row)[0, feature_index])
def fetch_data_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, group_by='column')
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Check ticker or dates.")
    # Flatten potential MultiIndex columns and select single ticker slice
    if isinstance(df.columns, pd.MultiIndex):
        # If last level contains the ticker, slice it
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            # Fallback: drop all but first level
            df.columns = ['_'.join([str(x) for x in col if x is not None]) for col in df.columns]
    # Normalize column names to Title-case expected
    col_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=col_map)
    # Some providers may give 'Adj Close' only; ensure core columns exist
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try to reconstruct Open/High/Low from Close if absolutely necessary
        if missing and 'Close' in df.columns:
            for c in missing:
                if c != 'Volume':
                    df[c] = df['Close']
        # If Volume missing, fill with zeros
        if 'Volume' in missing:
            df['Volume'] = 0.0
    df = df[required]
    df = df.dropna()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: sort, drop duplicates/nulls, handle infs, remove outliers by returns, winsorize tails.
    This keeps the price structure intact while mitigating extreme spikes.
    """
    cleaned = df.copy()
    # Order & de-duplicate
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
    # Replace infs and obvious bad values
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    # Forward/backward fill small gaps, then drop remaining
    cleaned = cleaned.ffill().bfill().dropna()
    # Compute simple returns for outlier detection
    returns = cleaned['Close'].pct_change()
    # IQR-based filter on returns (conservative whiskers = 3*IQR)
    ret_no_na = returns.dropna()
    q1, q3 = ret_no_na.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    mask = (returns.between(lower, upper)) | (returns.isna())
    cleaned = cleaned.loc[mask]
    # Winsorize numeric columns at 1st/99th percentile to limit remaining tails
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    lower_clip = cleaned[numeric_cols].quantile(0.01)
    upper_clip = cleaned[numeric_cols].quantile(0.99)
    cleaned[numeric_cols] = cleaned[numeric_cols].clip(lower=lower_clip, upper=upper_clip, axis=1)
    # Final drop of any gaps from filtering
    cleaned = cleaned.dropna()
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close']).diff()
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['EMA20'] = compute_ema(df['Close'], 20)
    macd_df = compute_macd(df['Close'])
    df = pd.concat([df, macd_df], axis=1)
    df['DayOfWeek'] = df.index.dayofweek
    df = df.dropna()
    return df


def prepare_data(ticker: str, start_date: str, end_date: str, sequence_length: int):
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    raw = fetch_data_yfinance(ticker, start_date, end_date)
    print(f"Fetched {len(raw)} rows")
    raw = preprocess_data(raw)
    print(f"After preprocessing: {len(raw)} rows")
    
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    df = engineer_features(raw)

    feature_cols = [
        'Open','High','Low','Close','Volume','RSI14','EMA20',
        'MACD','MACD_signal','MACD_hist','DayOfWeek','LogReturn'
    ]
    # Target: next-step log return of Close
    target_series = df['LogReturn'].shift(-1).dropna()

    # Align features and base prices with target index (t), predicting t+1 return
    df.columns = df.columns.astype(str)
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = sorted(list(set(feature_cols) - set(available_cols)))
    if missing:
        print(f"Warning: missing engineered columns skipped: {missing}")

    X_df = df.loc[target_series.index, available_cols]
    base_prices = df.loc[target_series.index, 'Close']  # Close at time t used to reconstruct t+1 price

    X = X_df.values.astype(np.float32)
    y = target_series.values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    base_seq = base_prices.values[sequence_length:]
    dates = target_series.index[sequence_length:]

    print(f"Sequences: {X_seq.shape} | Target: {y_seq.shape}")
    
    return {
        'X_seq': X_seq,
        'y_seq': y_seq,
        'dates': dates,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'base_seq': base_seq,
        'feature_cols': feature_cols,
        'close_feature_index': feature_cols.index('Close') if 'Close' in feature_cols else 0,
        'logret_feature_index': feature_cols.index('LogReturn') if 'LogReturn' in feature_cols else len(feature_cols)-1
    }


def split_data(X, y, dates, base_seq, train_split, val_split):
    n = len(X)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    dates_train = dates[:n_train]
    dates_val = dates[n_train:n_train+n_val]
    dates_test = dates[n_train+n_val:]
    base_train = base_seq[:n_train]
    base_val = base_seq[n_train:n_train+n_val]
    base_test = base_seq[n_train+n_val:]
    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    return {
        'train': (X_train, y_train, dates_train, base_train),
        'val': (X_val, y_val, dates_val, base_val),
        'test': (X_test, y_test, dates_test, base_test)
    }


# --------------------------
# Models
# --------------------------
def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.LSTM(64, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.Dense(1, dtype='float32')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


def build_gru(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.GRU(64, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.Dense(1, dtype='float32')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


def build_transformer(input_shape, num_heads=4, ff_dim=128, dropout=0.2):
    inp = layers.Input(shape=input_shape)
    x = inp
    # Positional encoding (simple learnable)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embed = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    x = x + pos_embed
    # Two encoder blocks
    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(input_shape[1])(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout + 0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, dtype='float32')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


def train_models(data_splits, input_shape, epochs, batch_size, selected_models=None, use_early_stopping=True):
    X_train, y_train, _, _ = data_splits['train']
    X_val, y_val, _, _ = data_splits['val']

    if selected_models is None:
        selected_models = ['LSTM', 'GRU', 'Transformer']

    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) if use_early_stopping else None
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5) if use_early_stopping else None
    # Cosine decay with warmup
    steps_per_epoch = max(1, len(data_splits['train'][0]) // batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.1 * total_steps))
    def lr_schedule(step):
        step = tf.cast(step, tf.float32)
        lr_base = tf.constant(CONFIG['learning_rate'], tf.float32)
        lr_min = lr_base * 0.1
        def warm():
            return lr_base * (step / tf.cast(warmup_steps, tf.float32))
        def cosine():
            progress = (step - warmup_steps) / tf.cast(max(1, total_steps - warmup_steps), tf.float32)
            return lr_min + (lr_base - lr_min) * 0.5 * (1 + tf.cos(np.pi * tf.clip_by_value(progress, 0.0, 1.0)))
        return tf.where(step < warmup_steps, warm(), cosine())
    lr_callback = callbacks.LearningRateScheduler(lambda s: float(lr_schedule(s).numpy()), verbose=0)

    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    models_dict = {}

    if 'LSTM' in selected_models:
        print("\nTraining LSTM...")
        lstm = build_lstm(input_shape)
        cbs = [cb for cb in [es, rlrop, lr_callback] if cb is not None]
        hist_lstm = lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=cbs)
        models_dict['LSTM'] = lstm
        models_dict['LSTM_history'] = hist_lstm.history

    if 'GRU' in selected_models:
        print("\nTraining GRU...")
        gru = build_gru(input_shape)
        cbs = [cb for cb in [es, rlrop, lr_callback] if cb is not None]
        hist_gru = gru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=cbs)
        models_dict['GRU'] = gru
        models_dict['GRU_history'] = hist_gru.history

    if 'Transformer' in selected_models:
        print("\nTraining Transformer...")
        transformer = build_transformer(input_shape)
        cbs = [cb for cb in [es, rlrop, lr_callback] if cb is not None]
        hist_tr = transformer.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=cbs)
        models_dict['Transformer'] = transformer
        models_dict['Transformer_history'] = hist_tr.history

    return models_dict


def evaluate_and_ensemble(models_dict, data_splits, scaler_y):
    print("\n" + "="*70)
    print("STEP 4: EVALUATION & ENSEMBLE")
    print("="*70)
    X_test, y_test, dates_test, base_test = data_splits['test']
    preds = {}
    metrics_all = {}
    # Inverse-transform true log-returns and reconstruct true prices for comparison
    logret_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_true = base_test * np.exp(logret_true)
    for name, mdl in models_dict.items():
        if name.endswith('_history'):
            continue
        y_pred_scaled = mdl.predict(X_test, verbose=0).ravel()
        # Inverse to returns and reconstruct price: price_{t+1} = price_t * exp(pred_logret)
        logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        # Align base_test (price_t) to y_pred length
        price_pred = base_test * np.exp(logret_pred)
        preds[name] = price_pred
        metrics_all[name] = calculate_metrics(y_true, price_pred)
        print(f"{name} -> RMSE: {metrics_all[name]['RMSE']:.4f} | MAE: {metrics_all[name]['MAE']:.4f} | DA: {metrics_all[name]['Directional_Accuracy']:.2f}%")

    # Simple weighted ensemble (equal weights). If multiple models are present we average.
    model_names = [k for k in models_dict.keys() if not k.endswith('_history')]
    y_stack = np.stack([preds[n] for n in model_names], axis=1)
    weights = np.ones(len(model_names), dtype=np.float32) / float(len(model_names))
    y_ens = (y_stack * weights).sum(axis=1)
    preds['Ensemble'] = y_ens
    metrics_all['Ensemble'] = calculate_metrics(y_true, y_ens)
    print(f"Ensemble -> RMSE: {metrics_all['Ensemble']['RMSE']:.4f} | MAE: {metrics_all['Ensemble']['MAE']:.4f} | DA: {metrics_all['Ensemble']['Directional_Accuracy']:.2f}%")

    # Confidence via model spread (std across models)
    std = y_stack.std(axis=1)
    lower = y_ens - 1.96 * std
    upper = y_ens + 1.96 * std
    
    return {
        'predictions': preds,
        'metrics': metrics_all,
        'y_test_actual': y_true,
        'dates_test': dates_test,
        'confidence_intervals': (lower, upper)
    }


# --------------------------
# Backtesting (simple sign strategy)
# --------------------------
def run_backtest(pred, actual, dates, initial_capital: float):
    pred_shift = np.roll(pred, 1)
    pred_shift[0] = pred_shift[1]
    signal = np.sign(pred - pred_shift)
    ret = np.concatenate([[0.0], np.diff(actual) / actual[:-1]])
    strat_ret = ret * signal
    equity = initial_capital * (1 + strat_ret).cumprod()
    buy_signals = np.where(signal > 0, actual, np.nan)
    sell_signals = np.where(signal < 0, actual, np.nan)
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-9) * np.sqrt(252)
    total_return = (equity[-1] / equity[0] - 1) * 100
    buy_hold = (actual[-1] / actual[0] - 1) * 100
    return {
        'equity': equity,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'metrics': {
            'Sharpe Ratio': sharpe,
            'Total Return (%)': total_return,
            'Buy & Hold Return (%)': buy_hold
        }
    }


def plot_trading_signals(dates, prices, buy, sell):
    plt.figure(figsize=(12, 4))
    plt.plot(dates, prices, color='black', linewidth=1, label='Price')
    plt.scatter(dates, buy, marker='^', color='green', label='Buy', s=30)
    plt.scatter(dates, sell, marker='v', color='red', label='Sell', s=30)
    plt.legend(); plt.tight_layout(); plt.show()


# --------------------------
# Orchestration
# --------------------------
def main():
    print("\n" + "="*70)
    print("STOCK PRICE PREDICTION WITH DEEP LEARNING")
    print(f"Ticker: {CONFIG['ticker']}")
    print(f"Period: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print("="*70)
    
    data = prepare_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'], CONFIG['sequence_length'])
    splits = split_data(data['X_seq'], data['y_seq'], data['dates'], CONFIG['train_split'], CONFIG['val_split'])
    input_shape = (CONFIG['sequence_length'], data['X_seq'].shape[2])
    models_dict = train_models(splits, input_shape, CONFIG['epochs'], CONFIG['batch_size'])
    eval_results = evaluate_and_ensemble(models_dict, splits, data['scaler_y'])

    print("\n" + "="*70)
    print("STEP 5: BACKTESTING")
    print("="*70)
    backtest = run_backtest(eval_results['predictions']['Ensemble'], eval_results['y_test_actual'], eval_results['dates_test'], CONFIG['initial_capital'])

    print("\n" + "="*70)
    print("STEP 6: VISUALIZATION")
    print("="*70)
    set_plot_style()
    print("\nModel metrics (lower RMSE is better):")
    print(create_metrics_table(eval_results['metrics']).to_string())
    plot_predictions(eval_results['dates_test'], eval_results['y_test_actual'], eval_results['predictions'], title=f"{CONFIG['ticker']} Predictions")
    plot_confidence_intervals(eval_results['dates_test'], eval_results['y_test_actual'], eval_results['predictions']['Ensemble'], eval_results['confidence_intervals'])
    plot_trading_signals(eval_results['dates_test'], eval_results['y_test_actual'], backtest['buy_signals'], backtest['sell_signals'])

    print("\nBacktest metrics:")
    for k, v in backtest['metrics'].items():
        print(f"{k}: {v:.2f}")


if __name__ == '__main__':
    import sys
    # If started with `python app.py serve`, run API server; otherwise run CLI pipeline
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"serve", "server", "api"}:
        app = Flask(__name__)
        CORS(app)

        @app.route('/api/health', methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @app.route('/api/predict', methods=['POST'])
        def api_predict():
            payload = request.get_json(force=True) or {}
            ticker = payload.get('ticker', CONFIG['ticker']).upper()
            start = payload.get('start_date', CONFIG['start_date'])
            end = payload.get('end_date', CONFIG['end_date'])
            seq_len = int(payload.get('sequence_length', CONFIG['sequence_length']))
            epochs = int(payload.get('epochs', CONFIG['epochs']))
            batch_size = int(payload.get('batch_size', CONFIG['batch_size']))
            future_days = int(payload.get('future_days', 5))

            # Prepare, split, train, evaluate
            data = prepare_data(ticker, start, end, seq_len)
            splits = split_data(data['X_seq'], data['y_seq'], data['dates'], data['base_seq'], CONFIG['train_split'], CONFIG['val_split'])
            input_shape = (seq_len, data['X_seq'].shape[2])
            # Optional: train only selected model from frontend
            selected = payload.get('model')
            selected_list = [selected] if selected in {'LSTM','GRU','Transformer'} else None
            use_es = bool(payload.get('early_stopping', True))
            models_dict = train_models(splits, input_shape, epochs, batch_size, selected_models=selected_list, use_early_stopping=use_es)
            eval_results = evaluate_and_ensemble(models_dict, splits, data['scaler_y'])

            # Backtest
            backtest = run_backtest(eval_results['predictions']['Ensemble'], eval_results['y_test_actual'], eval_results['dates_test'], float(payload.get('initial_capital', CONFIG['initial_capital'])))

            # Multi-step forecast using last sequence; writes predicted Close back into the feature window correctly scaled
            def roll_forecast(models_dict, last_seq, steps, scaler_y, scaler_X, close_idx):
                seq = last_seq.copy().astype(np.float32)
                preds = []
                for _ in range(steps):
                    step_preds = []
                    for name, mdl in models_dict.items():
                        if name.endswith('_history'):
                            continue
                        yhat_scaled = mdl.predict(seq[np.newaxis, ...], verbose=0).ravel()[0]
                        yhat = scaler_y.inverse_transform([[yhat_scaled]])[0,0]
                        step_preds.append(yhat)
                    yhat_avg = float(np.mean(step_preds))
                    preds.append(yhat_avg)
                    # shift seq window by one and update only the Close feature at the last timestep with correctly scaled value
                    seq = np.roll(seq, -1, axis=0)
                    seq[-1, close_idx] = scale_single_feature(yhat_avg, scaler_X, close_idx)
                return preds

            last_seq = data['X_seq'][-1]
            future_forecast = roll_forecast(models_dict, last_seq, future_days, data['scaler_y'], data.get('scaler_X', data.get('scaler')), data['close_feature_index'])
            last_date = eval_results['dates_test'][-1]
            future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(future_days)]

            # Prepare JSON
            dates_iso = [d.strftime('%Y-%m-%d') for d in eval_results['dates_test']]
            # Replace NaNs with None for JSON safety
            def safe_scalar(v):
                if v is None:
                    return None
                if isinstance(v, (np.floating, np.integer)):
                    v = v.item()
                if isinstance(v, (float, int)):
                    if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        return None
                    return v
                return float(v)

            def safe_list(a):
                return [safe_scalar(v) for v in a]

            response = {
                'ticker': ticker,
                'dates': dates_iso,
                'actual': safe_list(eval_results['y_test_actual']),
                'predictions': {k: safe_list(v) for k, v in eval_results['predictions'].items()},
                'confidence': {
                    'lower': safe_list(eval_results['confidence_intervals'][0]),
                    'upper': safe_list(eval_results['confidence_intervals'][1])
                },
                'metrics': {k: {mk: safe_scalar(mv) for mk, mv in md.items()} for k, md in eval_results['metrics'].items()},
                'backtest': {
                    'equity': safe_list(backtest['equity']),
                    'buy_signals': safe_list(backtest['buy_signals']),
                    'sell_signals': safe_list(backtest['sell_signals']),
                    'metrics': {mk: safe_scalar(mv) for mk, mv in backtest['metrics'].items()}
                },
                'future': {
                    'dates': future_dates,
                    'predictions': safe_list(future_forecast)
                }
            }
            # Include compact training histories for progress visualization
            histories = {k.replace('_history',''): {hk: safe_list(hv) for hk, hv in h.items()} for k, h in models_dict.items() if k.endswith('_history')}
            response['histories'] = histories
            return jsonify(response)

        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        main()