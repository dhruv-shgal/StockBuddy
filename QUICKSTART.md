# 🚀 Quick Start Guide - StockBuddy

Get up and running with StockBuddy's Live Prediction Engine in 3 minutes!

## Step 1: Start the Application (30 seconds)

### Windows
Double-click `start.bat`

This will:
- ✅ Start Flask API backend (port 5000)
- ✅ Start HTML frontend (port 8080)
- ✅ Open your browser automatically

You should see two command windows open:
1. **Flask API Backend** - Shows "Running on http://127.0.0.1:5000"
2. **HTML Frontend** - Shows "Serving HTTP on 0.0.0.0 port 8080"

## Step 2: Make Your First Prediction (2 minutes)

### In the Browser

1. **Navigate to Predict Section**
   - Click "Predict" in the navigation menu
   - Or scroll down to "Live Prediction Engine"

2. **Use Default Settings** (Already configured for quick test)
   - Ticker: AAPL ✓
   - Model: LSTM ✓
   - Dates: 2020-01-01 to Today ✓
   - Sequence: 90 ✓
   - Epochs: 20 ✓
   - Batch: 16 ✓
   - Future: 5 days ✓

3. **Click "Train & Predict"**
   - Watch the progress bar
   - Read the status log
   - Wait 2-3 minutes for training

4. **View Results**
   - See performance metrics (RMSE, MAE, Accuracy, Sharpe)
   - Explore the interactive chart
   - Check future price predictions

## Step 3: Experiment (Optional)

### Try Different Stocks
```
Popular stocks to try:
- AAPL (Apple)
- GOOGL (Google)
- TSLA (Tesla)
- MSFT (Microsoft)
- NVDA (NVIDIA)
- AMZN (Amazon)
```

### Try Different Models
- **LSTM**: Best all-around (default)
- **GRU**: Faster training
- **Transformer**: Most advanced

### Adjust Parameters
- **Quick Test**: Epochs = 10, Sequence = 60
- **High Accuracy**: Epochs = 30, Sequence = 120

## Troubleshooting

### ❌ Browser doesn't open automatically
**Solution**: Manually open http://localhost:8080

### ❌ "Failed to generate prediction" error
**Solution**: 
1. Check Flask API window is running
2. Visit http://localhost:5000/api/health
3. Should see: `{"status": "ok"}`
4. If not, restart `start.bat`

### ❌ Training takes too long
**Solution**: Reduce epochs to 10 and sequence to 60

### ❌ Port already in use
**Solution**: 
1. Close other applications using ports 5000 or 8080
2. Restart `start.bat`

## Testing the API

### Quick API Test
1. Open `test_api.html` in your browser
2. Click "Test Health Endpoint"
3. Should see: ✓ Success! {"status": "ok"}
4. Click "Test Prediction Endpoint" for full test

### Manual API Test
Open browser and visit:
```
http://localhost:5000/api/health
```
Should return:
```json
{"status": "ok"}
```

## Understanding Results

### Good Prediction Indicators
- ✅ Directional Accuracy > 55%
- ✅ Sharpe Ratio > 1.0
- ✅ Narrow confidence bands on chart
- ✅ RMSE < 5% of stock price

### Warning Signs
- ⚠️ Directional Accuracy < 50%
- ⚠️ Sharpe Ratio < 0.5
- ⚠️ Very wide confidence bands
- ⚠️ Predictions don't follow trend

## Next Steps

### Learn More
- Read **PREDICTION_GUIDE.md** for detailed instructions
- Check **WHATS_NEW.md** for feature overview
- Review **README.md** for technical details

### Experiment
- Try different stocks
- Compare models (LSTM vs GRU vs Transformer)
- Adjust training parameters
- Test different date ranges

### Advanced Usage
- Increase epochs for better accuracy
- Use longer sequences for long-term trends
- Try ensemble predictions
- Backtest different strategies

## Important Reminders

⚠️ **Not Financial Advice**: Educational purposes only

⚠️ **Past Performance**: Doesn't guarantee future results

⚠️ **Do Your Research**: Always verify predictions

⚠️ **Use Responsibly**: Understand the risks

## Support

### Need Help?
1. Check the troubleshooting section above
2. Review PREDICTION_GUIDE.md
3. Test API with test_api.html
4. Check browser console (F12) for errors
5. Review Flask API terminal output

### Common Questions

**Q: How long does training take?**
A: 2-5 minutes depending on parameters

**Q: Can I use any stock ticker?**
A: Yes, any valid ticker from Yahoo Finance

**Q: Which model is best?**
A: Start with LSTM, it's a good all-around choice

**Q: How accurate are predictions?**
A: Varies by stock, typically 55-70% directional accuracy

**Q: Can I save my predictions?**
A: Currently no, but you can screenshot or export the chart

## Quick Reference

### Default Configuration
```
Ticker: AAPL
Model: LSTM
Start Date: 2020-01-01
End Date: Today
Sequence Length: 90
Epochs: 20
Batch Size: 16
Future Days: 5
```

### Fast Configuration (1-2 min)
```
Ticker: AAPL
Model: GRU
Start Date: 2023-01-01
End Date: Today
Sequence Length: 60
Epochs: 10
Batch Size: 16
Future Days: 3
```

### Accurate Configuration (5-8 min)
```
Ticker: AAPL
Model: Transformer
Start Date: 2019-01-01
End Date: Today
Sequence Length: 120
Epochs: 30
Batch Size: 32
Future Days: 10
```

## Success Checklist

- [ ] start.bat executed successfully
- [ ] Two command windows are open
- [ ] Browser opened to localhost:8080
- [ ] Can see StockBuddy website
- [ ] Navigated to Predict section
- [ ] Clicked "Train & Predict"
- [ ] Saw progress bar and status log
- [ ] Results displayed with chart
- [ ] Metrics and future predictions visible

## You're Ready! 🎉

Start making predictions and exploring the power of AI-driven stock analysis!

---

**Need more help?** Check out PREDICTION_GUIDE.md for comprehensive documentation.