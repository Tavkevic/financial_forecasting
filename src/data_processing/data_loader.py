import pandas as pd
import numpy as np
from typing import Tuple, Dict
import yaml

class FinancialDataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_price_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка и предобработка ценовых данных"""
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def load_news_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка и предобработка новостных данных"""
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def merge_datasets(self, price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Объединение ценовых и новостных данных"""
        # Агрегация новостей по дням
        news_agg = self._aggregate_news_features(news_df)
        
        # Объединение с ценовыми данными
        merged_df = price_df.merge(news_agg, on='date', how='left')
        merged_df = merged_df.fillna(method='ffill')
        
        return merged_df
    
    def _aggregate_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация новостных признаков"""
        # Расчет sentiment scores
        news_df['sentiment_score'] = news_df['content'].apply(self._calculate_sentiment)
        
        # Агрегация по дням
        daily_news = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'title': 'count'
        }).reset_index()
        
        daily_news.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                             'news_count', 'title_count']
        
        return daily_news
    
    def _calculate_sentiment(self, text: str) -> float:
        """Упрощенный расчет тональности текста"""
        # Здесь будет более сложная логика с использованием NLP
        positive_words = ['рост', 'прибыль', 'успех', 'позитивный']
        negative_words = ['падение', 'убыток', 'риск', 'негативный']
        
        if not isinstance(text, str):
            return 0.0
            
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)