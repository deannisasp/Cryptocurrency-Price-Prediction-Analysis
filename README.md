# Cryptocurrency Price Prediction & Analysis 📊

A comprehensive Python project for cryptocurrency price prediction using machine learning and technical analysis visualization.

## Features ✨

- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Machine Learning Models**: 
  - Random Forest Classifier
  - XGBoost Classifier
  - Support Vector Machine (SVM)
- **Interactive Visualizations**: Dark-themed trading charts with multiple indicators
- **Statistical Analysis**: 24h price changes, volume analysis, trend detection

## Installation 🚀

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-analysis.git
cd crypto-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

### Basic Analysis

```python
python crypto_analysis.py
```

The program will:
1. Load cryptocurrency data from `data.xlsx`
2. Display available cryptocurrencies
3. Prompt you to select a coin for analysis
4. Generate trading charts with technical indicators
5. Optionally train machine learning models for price prediction

### Data Format

Your `data.xlsx` file should contain the following columns:
- `coin`: Name of the cryptocurrency
- `date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `marketCap`: Market capitalization

Example:
```
coin     | date       | open     | high     | low      | close    | volume      | marketCap
---------|------------|----------|----------|----------|----------|-------------|------------
Bitcoin  | 2024-01-01 | 45000.00 | 46000.00 | 44500.00 | 45800.00 | 1000000000  | 900000000000
```

## Technical Indicators 📈

The project calculates and visualizes the following technical indicators:

### Moving Averages
- **SMA (Simple Moving Average)**: 5, 10, 20 periods
- **EMA (Exponential Moving Average)**: 12, 26 periods

### Momentum Indicators
- **RSI (Relative Strength Index)**: 14-period RSI
  - Overbought: > 70
  - Oversold: < 30
  
- **MACD (Moving Average Convergence Divergence)**
  - MACD Line
  - Signal Line
  - Histogram

### Volatility Indicators
- **Bollinger Bands**: 20-period with 2 standard deviations

## Machine Learning Models 🤖

### 1. Random Forest Classifier
- Ensemble method using multiple decision trees
- Best for handling non-linear relationships

### 2. XGBoost Classifier
- Gradient boosting algorithm
- Excellent for structured/tabular data

### 3. Support Vector Machine (SVM)
- Finds optimal hyperplane for classification
- Good for high-dimensional data

### Prediction Target
Models predict whether the price will go **UP** or **DOWN** in the next period.

## Visualization 📊

The trading chart includes:

1. **Price Chart** (Top Panel)
   - Closing price line
   - Moving averages (SMA5, SMA20)
   - Bollinger Bands
   - Volume bars overlay

2. **RSI Chart** (Middle Panel)
   - RSI line
   - Overbought/Oversold zones

3. **MACD Chart** (Bottom Panel)
   - MACD line
   - Signal line
   - Histogram

## Project Structure 📁

```
crypto-analysis/
│
├── crypto_analysis.py    # Main Python script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore          # Git ignore rules
├── LICENSE             # License file
│
└── data/
    └── data.xlsx       # Your cryptocurrency data (not included)
```

## Example Output 🖼️

```
================================================
📊 BITCOIN TRADING STATISTICS
================================================
💰 Current Price: $45,800.00
📈 24h Change: +$800.00 (+1.78%)
⬆️  24h High: $46,200.00
⬇️  24h Low: $44,500.00
📊 24h Volume: $1,250.50M
📈 RSI (14): 65.3 🟡 Neutral
📊 Trend: 🚀 Strong Bullish
================================================

================================================
🤖 TRAINING MODELS FOR BITCOIN
================================================
📊 Dataset size: 1200 rows
🎯 Features: 20

🌲 Training Random Forest...
✅ Random Forest Accuracy: 0.8542

🚀 Training XGBoost...
✅ XGBoost Accuracy: 0.8625

🎯 Training SVM...
✅ SVM Accuracy: 0.8458

================================================
📊 MODEL COMPARISON
================================================
Random Forest: 0.8542
XGBoost:       0.8625
SVM:           0.8458
================================================

🏆 Best Model: XGBOOST (Accuracy: 0.8625)
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer ⚠️

This tool is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. The predictions made by this tool should not be considered as financial advice. Always do your own research before making investment decisions.

## Author ✍️

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## Acknowledgments 🙏

- Thanks to the open-source community for the amazing libraries
- Inspired by various cryptocurrency analysis tools
- Special thanks to contributors and testers

---

**⭐ If you find this project useful, please consider giving it a star!**
