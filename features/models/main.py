import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

from data_processing.data_loader import FinancialDataLoader
from features.feature_engineering import FeatureEngineer
from models.hybrid_model import HybridForecastingModel
from utils.metrics import calculate_metrics

class FinancialForecastingPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.data_loader = FinancialDataLoader(config_path)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model = None
        self.scaler = StandardScaler()
    
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run_pipeline(self, price_data_path: str, news_data_path: str):
        """Запуск полного пайплайна"""
        
        print("1. Загрузка данных...")
        price_data = self.data_loader.load_price_data(price_data_path)
        news_data = self.data_loader.load_news_data(news_data_path)
        
        print("2. Объединение данных...")
        merged_data = self.data_loader.merge_datasets(price_data, news_data)
        
        print("3. Создание признаков...")
        features_data = self.feature_engineer.create_technical_indicators(merged_data)
        features_data = self.feature_engineer.create_news_features(features_data)
        
        # Удаление NaN значений
        features_data = features_data.dropna()
        
        print("4. Подготовка последовательностей...")
        X, y = self.feature_engineer.prepare_sequences(features_data)
        
        print("5. Масштабирование признаков...")
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        print("6. Разделение на train/val...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print("7. Обучение модели...")
        self.model = HybridForecastingModel(self.config)
        history = self.model.train(X_train, y_train, X_val, y_val)
        
        print("8. Оценка модели...")
        y_pred = self.model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        
        print("\nМетрики качества:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return history, metrics

if __name__ == "__main__":
    pipeline = FinancialForecastingPipeline("config/config.yaml")
    pipeline.run_pipeline("data/raw/prices.csv", "data/raw/news.csv")