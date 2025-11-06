# StockPred - AI-Powered Stock Price Prediction

A comprehensive stock price prediction platform using deep learning models (LSTM, GRU, Transformer) with a modern HTML frontend and Flask API backend.

## Quick Start

1. **Start the Application**
   ```bash
   start.bat
   ```
   This will:
   - Start the Flask API backend on port 5000
   - Start the HTML frontend on port 8080
   - Automatically open your browser to http://localhost:8080

2. **Use the Application**
   - The browser will automatically open to http://localhost:8080
   - Navigate to the "Predict" section
   - Enter a stock ticker (e.g., AAPL, GOOGL, TSLA)
   - Set start and end dates for training data
   - Choose a model (LSTM, GRU, or Transformer)
   - Configure training parameters (epochs, batch size, sequence length)
   - Click "Train & Predict" to start training
   - View real-time training progress and logs
   - Analyze predictions with interactive charts, confidence intervals, and metrics
   - See future price forecasts in a detailed table

## Features

- **Real-time Data Fetching**: Uses yfinance to get stock data
- **Multiple AI Models**: LSTM, GRU, and Transformer architectures
- **Live Prediction Engine**: Interactive web interface to train models and predict stock prices
- **Technical Indicators**: RSI, EMA, MACD calculations
- **Ensemble Predictions**: Combines multiple models for better accuracy
- **Confidence Intervals**: Shows prediction uncertainty
- **Backtesting**: Evaluates strategy performance with metrics
- **Interactive Charts**: Visualizes predictions and results with Chart.js
- **Future Forecasting**: Predicts next N days with detailed tables

## Architecture

- **Frontend**: Modern HTML5 with Chart.js for visualizations
- **Backend**: Flask API with TensorFlow/Keras for deep learning
- **Data**: yfinance for real-time stock data fetching
- **Models**: Custom LSTM, GRU, and Transformer implementations

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Train models and get predictions

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- yfinance
- pandas, numpy, scikit-learn
- matplotlib (for backend plotting)

## Troubleshooting

If you encounter issues:
1. Ensure all Python dependencies are installed
2. Check that ports 5000 and 8080 are available
3. Verify TensorFlow is working: `python -c "import tensorflow; print(tensorflow.__version__)"`
4. Check the console output in both command windows for error messages

## Usage Tips

- Use recent date ranges (last 2-5 years) for better model performance
- LSTM works well for longer sequences, GRU is faster
- Transformer model may take longer to train but can capture complex patterns
- Check the confidence intervals to understand prediction uncertainty