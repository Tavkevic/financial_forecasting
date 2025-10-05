import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class HybridForecastingModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
    
    def build_model(self, sequence_length: int, n_features: int) -> Model:
        """Построение гибридной модели LSTM"""
        
        # Вход для временных рядов
        ts_input = Input(shape=(sequence_length, n_features), name='time_series_input')
        
        # LSTM слои
        x = LSTM(self.config['model']['lstm_units'][0], return_sequences=True)(ts_input)
        x = BatchNormalization()(x)
        x = Dropout(self.config['model']['dropout_rate'])(x)
        
        x = LSTM(self.config['model']['lstm_units'][1], return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config['model']['dropout_rate'])(x)
        
        # Дополнительные плотные слои
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config['model']['dropout_rate'])(x)
        
        x = Dense(32, activation='relu')(x)
        
        # Выходной слой
        output = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=ts_input, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['model']['learning_rate']),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
        """Обучение модели"""
        
        if self.model is None:
            self.model = self.build_model(X_train.shape[1], X_train.shape[2])
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Прогнозирование"""
        return self.model.predict(X)
    