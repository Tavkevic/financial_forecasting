import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание технических индикаторов"""
        df = df.copy()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        return df
    
    def create_news_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков из новостей"""
        df = df.copy()
        
        # Lag features для новостей
        for lag in [1, 2, 3]:
            df[f'sentiment_lag_{lag}'] = df['sentiment_mean'].shift(lag)
            df[f'news_count_lag_{lag}'] = df['news_count'].shift(lag)
        
        # Rolling statistics для sentiment
        df['sentiment_rolling_mean_7'] = df['sentiment_mean'].rolling(7).mean()
        df['sentiment_rolling_std_7'] = df['sentiment_mean'].rolling(7).std()
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка последовательностей для LSTM"""
        feature_columns = [col for col in df.columns if col not in ['date', target_column]]
        
        X, y = [], []
        
        for i in range(len(df) - self.config['data']['sequence_length'] - self.config['data']['forecast_horizon']):
            # Признаки
            sequence = df[feature_columns].iloc[i:i + self.config['data']['sequence_length']].values
            # Целевая переменная (цена через N дней)
            target = df[target_column].iloc[i + self.config['data']['sequence_length'] + self.config['data']['forecast_horizon'] - 1]
            
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)