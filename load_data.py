import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, ticker="^IBEX", start_date="2000-01-01", end_date="2024-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler()
        
    def download_data(self):
        """Función que descarga datos históricos de Yahoo Finance."""
        print(f"Descargando datos para {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Limpieza básica
        df.dropna(inplace=True)
        self.data = df
        return self.data

    def add_features(self):
        """Añade indicadores técnicos y retornos logarítmicos."""
        # 1. Retornos logarítmicos (más estables para redes neuronales)
        self.data['Log_Ret'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # 2. Volatilidad (Desviación estándar móvil de 20 días)
        self.data['Volatilidad'] = self.data['Log_Ret'].rolling(window=20).std()
        
        # 3. Indicadores Técnicos usando pandas_ta
        self.data.ta.rsi(length=14, append=True)
        self.data.ta.macd(append=True)
        self.data.ta.sma(length=50, append=True)
        self.data.ta.sma(length=200, append=True)
        
        # Limpiar NaNs producidos por las medias móviles
        self.data.dropna(inplace=True)
        print(f"Características añadidas. Columnas actuales: {self.data.columns.tolist()}")

    def split_data(self, stability_end="2019-12-31"):
        """
        Divide los datos en entrenamiento (estabilidad) y test (crisis/anomalía).
        El Autoencoder SOLO verá el periodo de estabilidad.
        """
        train_data = self.data[:stability_end]
        test_data = self.data[stability_end:]
        return train_data, test_data

    def create_sequences(self, data, window_size=30):
        """
        Transforma el DataFrame en tensores 3D para la CNN/TCN:
        (Número de muestras, Tamaño de ventana, Número de características)
        """
        # Seleccionamos solo las columnas numéricas que queremos usar
        features = data.values
        X = []
        for i in range(len(features) - window_size):
            X.append(features[i : i + window_size])
        
        return np.array(X)

    def scale_data(self, train_data, test_data):
        """Escala los datos basándose SOLO en el set de entrenamiento."""
        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        return train_scaled, test_scaled

# Ejemplo de uso en VS Code:
if __name__ == "__main__":
    processor = DataProcessor()
    df = processor.download_data()
    processor.add_features()
    
    # Dividimos antes de escalar para no "contaminar" el entrenamiento con datos del COVID
    train_df, test_df = processor.split_data(stability_end="2019-12-31")
    
    # Escalado
    train_scaled, test_scaled = processor.scale_data(train_df, test_df)
    
    # Creación de secuencias para la CNN (ventana de 30 días)
    X_train = processor.create_sequences(train_scaled, window_size=30)
    X_test = processor.create_sequences(test_scaled, window_size=30)
    
    print(f"Forma del set de entrenamiento: {X_train.shape}") # (muestras, 30, n_features)
    print(f"Forma del set de test: {X_test.shape}")