import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app import (
    CONFIG,
    prepare_data,
    split_data,
    train_models,
    evaluate_and_ensemble,
    run_backtest,
    set_plot_style,
    plot_predictions,
    plot_confidence_intervals,
    plot_trading_signals,
)

st.set_page_config(page_title="StockPred - Streamlit", layout="wide")

st.title("StockPred - Streamlit")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value=CONFIG['ticker'])
    start_default = datetime.date(2020, 1, 1)
    start = st.date_input("Start Date", value=start_default)
    end = st.date_input("End Date", value=datetime.date.fromisoformat(CONFIG['end_date']))
    model_choice = st.selectbox("Model", ["LSTM", "GRU", "Transformer"])
    seq_len = st.number_input("Sequence Length", min_value=30, max_value=365, value=CONFIG['sequence_length'], step=5)
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=CONFIG['epochs'])
    batch = st.number_input("Batch Size", min_value=8, max_value=256, value=CONFIG['batch_size'], step=8)
    future_days = st.number_input("Future Days", min_value=1, max_value=60, value=5)
    show_ensemble = st.checkbox("Show Ensemble on chart", value=False)
    train_btn = st.button("Train & Predict")

st.markdown("---")

if train_btn:
    with st.status("Fetching and preparing data...", expanded=True) as status:
        data = prepare_data(ticker, str(start), str(end), int(seq_len))
        st.write("Sequences:", data['X_seq'].shape)
        splits = split_data(
            data['X_seq'],
            data['y_seq'],
            data['dates'],
            data['base_seq'],
            CONFIG['train_split'],
            CONFIG['val_split']
        )
        status.update(label="Training model(s)...")
        models_dict = train_models(splits, (int(seq_len), data['X_seq'].shape[2]), int(epochs), int(batch), selected_models=[model_choice], use_early_stopping=True)
        status.update(label="Evaluating & ensembling...")
        eval_results = evaluate_and_ensemble(models_dict, splits, data['scaler_y'])
        status.update(label="Backtesting...")
        backtest = run_backtest(eval_results['predictions']['Ensemble'], eval_results['y_test_actual'], eval_results['dates_test'], CONFIG['initial_capital'])
        status.update(label="Done", state="complete")

    # Metrics table
    st.subheader("Metrics")
    metrics_df = pd.DataFrame(eval_results['metrics']).T
    st.dataframe(metrics_df.style.format({
        'RMSE': '{:.4f}',
        'MAE': '{:.4f}',
        'MAPE': '{:.2f}',
        'R2': '{:.4f}',
        'Directional_Accuracy': '{:.2f}'
    }))

    # Charts
    st.subheader("Predictions")
    # Plot predictions (explicit fig)
    set_plot_style()
    dates = eval_results['dates_test']
    y_true = eval_results['y_test_actual']
    preds_dict = eval_results['predictions']
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(dates, y_true, label='Actual', color='black', linewidth=2, zorder=3)
    # Draw selected model in red so it is always clearly visible
    selected_color = '#ef4444'  # red
    if model_choice in preds_dict:
        ax1.plot(dates, preds_dict[model_choice], label=model_choice, color=selected_color, linewidth=2.2)
    else:
        # Fallback to first available model prediction
        for key in ['LSTM','GRU','Transformer']:
            if key in preds_dict:
                ax1.plot(dates, preds_dict[key], label=key, color=selected_color, linewidth=2.2)
                break
    # Optionally draw ensemble
    if show_ensemble and 'Ensemble' in preds_dict:
        ax1.plot(dates, preds_dict['Ensemble'], label='Ensemble', color='#22c55e', linewidth=2.0, alpha=0.9)
    ax1.set_title(f"{ticker} Predictions")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

    # Confidence intervals (explicit fig)
    lower, upper = eval_results['confidence_intervals']
    y_ens = preds_dict['Ensemble']
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(dates, y_true, label='Actual', color='black')
    ax2.plot(dates, y_ens, label='Ensemble', color='green')
    ax2.fill_between(dates, lower, upper, color='green', alpha=0.15, label='Confidence band')
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("Backtest")
    st.metric("Sharpe Ratio", f"{backtest['metrics']['Sharpe Ratio']:.2f}")
    st.metric("Total Return (%)", f"{backtest['metrics']['Total Return (%)']:.2f}")
    st.metric("Buy & Hold Return (%)", f"{backtest['metrics']['Buy & Hold Return (%)']:.2f}")

    # Trading signals (explicit fig)
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(dates, y_true, color='black', linewidth=1, label='Price')
    ax3.scatter(dates, backtest['buy_signals'], marker='^', color='green', label='Buy', s=30)
    ax3.scatter(dates, backtest['sell_signals'], marker='v', color='red', label='Sell', s=30)
    ax3.legend()
    fig3.tight_layout()
    st.pyplot(fig3)

    # Future forecast (rolling): model predicts next-step log return → reconstruct next price
    st.subheader("Future Forecast")
    last_seq = data['X_seq'][-1]
    last_price = float(eval_results['y_test_actual'][-1])
    close_idx = int(data['close_feature_index'])
    preds_price = []
    mdl = [m for k,m in models_dict.items() if not k.endswith('_history')][0]
    for _ in range(int(future_days)):
        yhat_scaled = mdl.predict(last_seq[np.newaxis, ...], verbose=0).ravel()[0]
        logret = data['scaler_y'].inverse_transform([[yhat_scaled]])[0,0]
        next_price = last_price * float(np.exp(logret))
        preds_price.append(next_price)
        # Update window: shift and set Close feature scaled for next step
        last_seq = np.roll(last_seq, -1, axis=0)
        # scale single feature using scaler_X column stats
        sx = data['scaler_X']
        if hasattr(sx, 'data_min_') and hasattr(sx, 'scale_'):
            scaled_close = (next_price - sx.data_min_[close_idx]) * sx.scale_[close_idx]
        elif hasattr(sx, 'mean_') and hasattr(sx, 'scale_'):
            scaled_close = (next_price - sx.mean_[close_idx]) / (sx.scale_[close_idx] + 1e-12)
        else:
            row = np.zeros((1, last_seq.shape[1]), dtype=np.float32); row[0, close_idx] = next_price
            scaled_close = float(sx.transform(row)[0, close_idx])
        last_seq[-1, close_idx] = scaled_close
        last_price = next_price

    last_date = eval_results['dates_test'][-1]
    fut_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(int(future_days))]
    fut_df = pd.DataFrame({'Date': fut_dates, 'Predicted Price': preds_price})
    st.table(fut_df)


