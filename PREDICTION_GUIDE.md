# StockBuddy Prediction Guide

## Overview

The Live Prediction Engine allows you to train AI models on real stock market data and generate predictions with confidence intervals directly from your browser.

## Getting Started

### 1. Start the Application

Run the `start.bat` file which will:
- Start the Flask API backend on port 5000
- Start the HTML frontend on port 8080
- Automatically open your browser

### 2. Navigate to Prediction Section

Click on "Predict" in the navigation menu or scroll to the "Live Prediction Engine" section.

## Configuration Parameters

### Stock Ticker
- **What it is**: The stock symbol you want to predict (e.g., AAPL, GOOGL, TSLA)
- **Tip**: Use major stocks with good historical data for best results

### AI Model
Choose from three architectures:
- **LSTM (Long Short-Term Memory)**: Best for capturing long-term dependencies, good all-around choice
- **GRU (Gated Recurrent Unit)**: Faster training, efficient for shorter sequences
- **Transformer**: Attention-based, excellent for complex patterns but slower to train

### Date Range
- **Start Date**: Beginning of training data (recommend at least 2 years of history)
- **End Date**: End of training data (defaults to today)
- **Tip**: More data generally leads to better predictions, but takes longer to train

### Training Parameters

#### Sequence Length (30-365)
- Number of previous days the model looks at to make predictions
- **Default**: 90 days
- **Shorter (30-60)**: Faster training, captures recent trends
- **Longer (90-180)**: Better for long-term patterns, slower training

#### Training Epochs (5-100)
- Number of times the model trains on the entire dataset
- **Default**: 20 epochs
- **Fewer (5-10)**: Quick testing, may underfit
- **More (30-50)**: Better accuracy, longer training time
- **Too many (>50)**: Risk of overfitting

#### Batch Size (8-128)
- Number of samples processed before updating the model
- **Default**: 16
- **Smaller (8-16)**: More frequent updates, better for small datasets
- **Larger (32-64)**: Faster training, needs more memory

#### Future Days (1-60)
- How many days ahead to forecast
- **Default**: 5 days
- **Tip**: Predictions become less reliable further into the future

## Using the Prediction Engine

### Step 1: Configure Your Prediction
1. Enter the stock ticker (e.g., AAPL)
2. Select your preferred AI model
3. Set the date range for training data
4. Adjust training parameters as needed
5. Set how many future days to predict

### Step 2: Train & Predict
1. Click the "Train & Predict" button
2. Watch the progress bar and status log
3. Training typically takes 1-5 minutes depending on:
   - Amount of data
   - Number of epochs
   - Model complexity
   - Your computer's speed

### Step 3: Analyze Results

#### Metrics
- **RMSE (Root Mean Square Error)**: Lower is better, measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **Directional Accuracy**: Percentage of correct up/down predictions (>50% is good)
- **Sharpe Ratio**: Risk-adjusted return metric (>1.0 is good, >2.0 is excellent)

#### Prediction Chart
- **Gray Line**: Actual historical prices
- **Cyan Line**: Model predictions
- **Shaded Area**: Confidence interval (uncertainty range)
- **Tip**: Narrower confidence bands indicate more confident predictions

#### Future Forecast Table
- Shows predicted prices for the next N days
- Dates and predicted prices in an easy-to-read table
- **Remember**: These are predictions, not guarantees!

## Tips for Best Results

### Data Quality
- Use stocks with consistent trading history
- Avoid stocks with recent IPOs or major corporate events
- More historical data (3-5 years) generally improves accuracy

### Model Selection
- **Start with LSTM**: Good balance of accuracy and speed
- **Try GRU**: If LSTM is too slow
- **Use Transformer**: For complex patterns, if you have time

### Training Parameters
- **Quick Test**: 60 sequence length, 10 epochs, 16 batch size
- **Balanced**: 90 sequence length, 20 epochs, 16 batch size (default)
- **High Accuracy**: 120 sequence length, 30 epochs, 32 batch size

### Interpreting Results
- **Directional Accuracy >60%**: Good model performance
- **Sharpe Ratio >1.5**: Strong risk-adjusted returns
- **Narrow Confidence Bands**: More reliable predictions
- **Wide Confidence Bands**: Higher uncertainty, be cautious

## Troubleshooting

### "Failed to generate prediction" Error
**Solution**: Make sure the Flask API is running
- Check that the Flask API window is open
- Verify it shows "Running on http://127.0.0.1:5000"
- Try restarting with `start.bat`

### Training Takes Too Long
**Solutions**:
- Reduce the number of epochs (try 10-15)
- Use a shorter sequence length (try 60)
- Select a smaller date range (2-3 years instead of 5)
- Choose GRU instead of LSTM or Transformer

### Poor Prediction Accuracy
**Solutions**:
- Increase training epochs (try 30-40)
- Use more historical data (3-5 years)
- Try a different model architecture
- Increase sequence length (try 120-150)

### API Connection Error
**Solutions**:
- Ensure Flask backend is running on port 5000
- Check Windows Firewall isn't blocking the connection
- Try accessing http://localhost:5000/api/health in your browser
- Restart both backend and frontend

## API Testing

Use the included `test_api.html` file to verify your API connection:
1. Open `test_api.html` in your browser
2. Click "Test Health Endpoint" to verify API is running
3. Click "Test Prediction Endpoint" for a quick prediction test

## Advanced Usage

### Custom Parameters
You can experiment with different combinations:
- **Day Trading Focus**: Short sequence (30-45), recent data (6 months)
- **Long-term Investing**: Long sequence (120-180), extensive data (5+ years)
- **Volatile Stocks**: Higher epochs (30-40), larger batch size (32-64)

### Comparing Models
Train the same stock with different models to compare:
1. Train with LSTM, note the metrics
2. Train with GRU, compare results
3. Train with Transformer, see which performs best
4. Use the best-performing model for your predictions

## Important Disclaimers

⚠️ **Not Financial Advice**: This tool is for educational and research purposes only

⚠️ **Past Performance**: Historical accuracy doesn't guarantee future results

⚠️ **Market Volatility**: Unexpected events can make predictions inaccurate

⚠️ **Use Responsibly**: Always do your own research before making investment decisions

## Support

For issues or questions:
- Check the console log in your browser (F12)
- Review the Flask API terminal output
- Ensure all dependencies are installed
- Verify Python and TensorFlow are working correctly

## Next Steps

- Experiment with different stocks and parameters
- Compare predictions across multiple models
- Track prediction accuracy over time
- Integrate with your own trading strategies (at your own risk)

Happy Predicting! 📈