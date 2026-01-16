"""
Cryptocurrency Price Prediction and Analysis
=============================================
Program untuk memprediksi harga cryptocurrency menggunakan Machine Learning
dan visualisasi data trading dengan indikator teknikal.

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')


# ==================== DATA LOADING ====================

def load_data(file_path='data.xlsx'):
    """
    Load cryptocurrency data from Excel file
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file containing cryptocurrency data
    
    Returns:
    --------
    pd.DataFrame
        Loaded cryptocurrency data
    """
    try:
        data = pd.read_excel(file_path)
        print(f"✅ Data loaded successfully: {len(data)} rows")
        return data
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


# ==================== FEATURE ENGINEERING ====================

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLCV data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional technical indicators
    """
    df = df.copy()
    
    # Moving Averages
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Price momentum
    df['momentum'] = df['close'].diff(4)
    df['price_change'] = df['close'].pct_change()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    return df


def create_target_variable(df, forecast_days=1):
    """
    Create target variable for prediction (price will go up or down)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    forecast_days : int
        Number of days to forecast ahead
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with target variable
    """
    df = df.copy()
    df['future_price'] = df['close'].shift(-forecast_days)
    df['target'] = (df['future_price'] > df['close']).astype(int)
    return df


# ==================== MODEL TRAINING ====================

def prepare_features(df):
    """
    Prepare features for machine learning model
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with technical indicators
    
    Returns:
    --------
    tuple
        (features_df, feature_columns)
    """
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'SMA5', 'SMA10', 'SMA20', 'EMA12', 'EMA26',
        'RSI14', 'MACD', 'MACD_signal', 'MACD_histogram',
        'BB_middle', 'BB_upper', 'BB_lower',
        'momentum', 'price_change', 'volume_sma'
    ]
    
    # Remove rows with NaN values
    df_clean = df.dropna(subset=feature_columns + ['target'])
    
    return df_clean, feature_columns


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : testing data
    
    Returns:
    --------
    model, accuracy, report
    """
    print("\n🌲 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"✅ Random Forest Accuracy: {accuracy:.4f}")
    
    return rf_model, accuracy, report


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : testing data
    
    Returns:
    --------
    model, accuracy, report
    """
    print("\n🚀 Training XGBoost...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"✅ XGBoost Accuracy: {accuracy:.4f}")
    
    return xgb_model, accuracy, report


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train SVM model
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : testing data
    
    Returns:
    --------
    model, accuracy, report
    """
    print("\n🎯 Training SVM...")
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"✅ SVM Accuracy: {accuracy:.4f}")
    
    return svm_model, accuracy, report


# ==================== VISUALIZATION ====================

def generate_sample_data(data, coin_name):
    """
    Generate sample data for a specific coin with technical indicators
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full cryptocurrency dataset
    coin_name : str
        Name of the cryptocurrency
    
    Returns:
    --------
    pd.DataFrame
        Filtered and processed data for the coin
    """
    # Filter data untuk koin tertentu
    df = data[data['coin'] == coin_name].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Set date as index for plotting
    df.set_index('date', inplace=True)
    
    return df


def create_trading_chart(df, coin_name='Cryptocurrency'):
    """
    Create comprehensive trading chart with technical indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV and technical indicators
    coin_name : str
        Name of the cryptocurrency
    
    Returns:
    --------
    matplotlib.figure.Figure
        Trading chart figure
    """
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                         gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # === SUBPLOT 1: Price and Moving Averages ===
    current_price = df['close'].iloc[-1]
    
    # Plot price
    ax1.plot(df.index, df['close'], color='#00d4ff', linewidth=2.5, 
             label=f'Price (${current_price:.2f})', zorder=3)
    
    # Plot moving averages
    if 'SMA5' in df.columns:
        ax1.plot(df.index, df['SMA5'], color='#ffd700', linewidth=1.5, 
                alpha=0.8, label='SMA 5', linestyle='--')
    if 'SMA20' in df.columns:
        ax1.plot(df.index, df['SMA20'], color='#ff6b6b', linewidth=1.5, 
                alpha=0.8, label='SMA 20', linestyle='--')
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
        ax1.plot(df.index, df['BB_upper'], color='#888888', linewidth=1, 
                alpha=0.5, linestyle=':')
        ax1.plot(df.index, df['BB_lower'], color='#888888', linewidth=1, 
                alpha=0.5, linestyle=':')
        ax1.fill_between(df.index, df['BB_upper'], df['BB_lower'], 
                         alpha=0.1, color='#888888')
    
    # Highlight current price
    ax1.axhline(y=current_price, color='#f85149', linestyle='--', 
                alpha=0.5, linewidth=1.5)
    ax1.text(df.index[-1], current_price, f'${current_price:.2f}', 
             color='#f85149', fontweight='bold', ha='right', va='bottom')
    
    # Format title and labels
    ax1.set_title(f'{coin_name} • 1d', fontsize=18, color='white', 
                  pad=20, fontweight='bold')
    ax1.legend(loc='upper left', frameon=False, fontsize=10)
    ax1.grid(True, alpha=0.2, color='#333333', linewidth=0.5)
    ax1.set_ylabel('Price', color='white', fontsize=12)
    
    # Format y-axis for price
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
    ax1.tick_params(axis='y', colors='white', labelsize=10)
    ax1.tick_params(axis='x', colors='white', labelsize=10)
    
    # Volume bars as overlay
    ax1_vol = ax1.twinx()
    volume_colors = ['#00ff88' if df['close'].iloc[i] >= df['open'].iloc[i] 
                    else '#ff4757' for i in range(len(df))]
    
    bars = ax1_vol.bar(df.index, df['volume'], alpha=0.4, 
                       color=volume_colors, width=1)
    ax1_vol.set_ylabel('Volume', color='white', fontsize=12)
    ax1_vol.tick_params(axis='y', colors='white', labelsize=10)
    
    # Format volume axis
    ax1_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K')
    )
    
    # === SUBPLOT 2: RSI ===
    if 'RSI14' in df.columns:
        ax2.plot(df.index, df['RSI14'], color='#a855f7', linewidth=2, label='RSI')
        ax2.axhline(y=70, color='#ff4757', linestyle='--', alpha=0.7, 
                   linewidth=1, label='Overbought (70)')
        ax2.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7, 
                   linewidth=1, label='Oversold (30)')
        ax2.fill_between(df.index, 30, 70, alpha=0.1, color='#666666')
        
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI', color='white', fontsize=12)
        ax2.legend(loc='upper left', frameon=False, fontsize=9)
        ax2.grid(True, alpha=0.2, color='#333333', linewidth=0.5)
        ax2.tick_params(axis='y', colors='white', labelsize=10)
        ax2.tick_params(axis='x', colors='white', labelsize=10)
    
    # === SUBPLOT 3: MACD ===
    if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        # MACD histogram
        colors = ['#00ff88' if x >= 0 else '#ff4757' for x in df['MACD_histogram']]
        ax3.bar(df.index, df['MACD_histogram'], color=colors, alpha=0.7, 
               width=1, label='Histogram')
        
        # MACD lines
        ax3.plot(df.index, df['MACD'], color='#ff9f43', linewidth=2, label='MACD')
        ax3.plot(df.index, df['MACD_signal'], color='#0abde3', linewidth=2, 
                label='Signal')
        
        ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=1)
        ax3.set_ylabel('MACD', color='white', fontsize=12)
        ax3.legend(loc='upper left', frameon=False, fontsize=9)
        ax3.grid(True, alpha=0.2, color='#333333', linewidth=0.5)
        ax3.tick_params(axis='y', colors='white', labelsize=10)
        ax3.tick_params(axis='x', colors='white', labelsize=10)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    
    # Only show x-axis labels on bottom subplot
    ax1.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='x', labelbottom=False)
    
    # Rotate x-axis labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Set figure background
    fig.patch.set_facecolor('#0a0a0a')
    
    return fig


def display_trading_stats(df, coin_name='Cryptocurrency'):
    """
    Display trading statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV and technical indicators
    coin_name : str
        Name of the cryptocurrency
    """
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    high_24h = df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()
    low_24h = df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()
    volume_24h = df['volume'].tail(24).sum() if len(df) >= 24 else df['volume'].sum()
    
    print("=" * 60)
    print(f"📊 {coin_name} TRADING STATISTICS")
    print("=" * 60)
    print(f"💰 Current Price: ${current_price:,.2f}")
    
    change_emoji = "📈" if price_change >= 0 else "📉"
    change_color = "+" if price_change >= 0 else ""
    print(f"{change_emoji} 24h Change: {change_color}${price_change:.2f} ({price_change_pct:+.2f}%)")
    
    print(f"⬆️  24h High: ${high_24h:,.2f}")
    print(f"⬇️  24h Low: ${low_24h:,.2f}")
    print(f"📊 24h Volume: ${volume_24h/1000000:.2f}M")
    
    # RSI status
    if 'RSI14' in df.columns:
        current_rsi = df['RSI14'].iloc[-1]
        if pd.notna(current_rsi):
            if current_rsi > 70:
                rsi_status = "🔴 Overbought"
            elif current_rsi < 30:
                rsi_status = "🟢 Oversold"
            else:
                rsi_status = "🟡 Neutral"
            print(f"📈 RSI (14): {current_rsi:.1f} {rsi_status}")
    
    # Simple trend analysis
    if all(col in df.columns for col in ['SMA5', 'SMA20']):
        sma_short = df['SMA5'].iloc[-1]
        sma_long = df['SMA20'].iloc[-1]
        
        if pd.notna(sma_short) and pd.notna(sma_long):
            if current_price > sma_short > sma_long:
                trend = "🚀 Strong Bullish"
            elif current_price > sma_long:
                trend = "📈 Bullish"
            elif current_price < sma_short < sma_long:
                trend = "🔻 Strong Bearish"
            else:
                trend = "📉 Bearish"
            print(f"📊 Trend: {trend}")
    
    print("=" * 60)


# ==================== MAIN FUNCTIONS ====================

def generate_prediction_visualization(data, coin_name='Bitcoin'):
    """
    Generate trading chart for a specific cryptocurrency
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full cryptocurrency dataset
    coin_name : str
        Name of the cryptocurrency
    
    Returns:
    --------
    tuple
        (df, fig) - DataFrame and Figure object
    """
    print(f"🔄 Generating trading chart for {coin_name}...")
    
    # Get data for selected coin
    df = generate_sample_data(data, coin_name)
    
    if df.empty:
        print(f"❌ No data found for {coin_name}")
        return None, None
    
    # Display stats
    display_trading_stats(df, coin_name)
    
    # Display chart
    fig = create_trading_chart(df, coin_name)
    
    # Show plot
    plt.show()
    
    return df, fig


def train_models(data, coin_name='Bitcoin', test_size=0.2):
    """
    Train multiple models for price prediction
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full cryptocurrency dataset
    coin_name : str
        Name of the cryptocurrency
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    dict
        Dictionary containing trained models and their accuracies
    """
    print(f"\n{'='*60}")
    print(f"🤖 TRAINING MODELS FOR {coin_name.upper()}")
    print(f"{'='*60}")
    
    # Filter data for specific coin
    df = data[data['coin'] == coin_name].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Create target variable
    df = create_target_variable(df, forecast_days=1)
    
    # Prepare features
    df_clean, feature_columns = prepare_features(df)
    
    print(f"📊 Dataset size: {len(df_clean)} rows")
    print(f"🎯 Features: {len(feature_columns)}")
    
    # Split data
    X = df_clean[feature_columns]
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    print(f"📝 Training set: {len(X_train)} samples")
    print(f"🧪 Testing set: {len(X_test)} samples")
    
    # Train models
    results = {}
    
    # Random Forest
    rf_model, rf_acc, rf_report = train_random_forest(X_train, y_train, X_test, y_test)
    results['random_forest'] = {
        'model': rf_model,
        'accuracy': rf_acc,
        'report': rf_report
    }
    
    # XGBoost
    xgb_model, xgb_acc, xgb_report = train_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {
        'model': xgb_model,
        'accuracy': xgb_acc,
        'report': xgb_report
    }
    
    # SVM
    svm_model, svm_acc, svm_report = train_svm(X_train, y_train, X_test, y_test)
    results['svm'] = {
        'model': svm_model,
        'accuracy': svm_acc,
        'report': svm_report
    }
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"XGBoost:       {xgb_acc:.4f}")
    print(f"SVM:           {svm_acc:.4f}")
    print(f"{'='*60}\n")
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"🏆 Best Model: {best_model_name.upper()} "
          f"(Accuracy: {results[best_model_name]['accuracy']:.4f})")
    
    return results


def main():
    """
    Main function to run the cryptocurrency analysis
    """
    print("\n" + "="*60)
    print("🚀 CRYPTOCURRENCY PRICE PREDICTION & ANALYSIS")
    print("="*60 + "\n")
    
    # Load data
    data = load_data('data.xlsx')
    
    if data is None:
        return
    
    # Display available coins
    print("\n📋 Available Cryptocurrencies:")
    available_coins = data['coin'].unique()
    for i, coin in enumerate(available_coins, 1):
        print(f"  {i}. {coin}")
    
    # Get user input
    selected_coin = input(f"\n💡 Enter cryptocurrency name to analyze: ").strip()
    
    if selected_coin not in available_coins:
        print(f"⚠️  '{selected_coin}' not found. Using first coin: {available_coins[0]}")
        selected_coin = available_coins[0]
    
    # Generate visualization
    df, fig = generate_prediction_visualization(data, selected_coin)
    
    if df is not None:
        print(f"\n✅ Visualization for {selected_coin} completed!")
        
        # Ask if user wants to train models
        train_choice = input("\n🤖 Do you want to train prediction models? (y/n): ").lower()
        
        if train_choice == 'y':
            results = train_models(data, selected_coin)
            
            # Display detailed report for best model
            best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            print(f"\n📋 Detailed Report for {best_model_name.upper()}:")
            print(results[best_model_name]['report'])
    
    print("\n" + "="*60)
    print("✨ Analysis Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
