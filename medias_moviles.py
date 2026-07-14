import pandas as pd
import numpy as np
from typing import Optional

from load_data import DataProcessor


def compute_sma(df: pd.DataFrame, column: str = 'Close', window: int = 30) -> pd.Series:
    """Calcula la media móvil simple (SMA).

    Args:
        df: DataFrame con datos de precios.
        column: Nombre de la columna sobre la que aplicar la media.
        window: Ventana en días.

    Returns:
        Serie con la SMA alineada al índice del DataFrame.
    """
    return df[column].rolling(window=window, min_periods=1).mean()


def compute_ema(df: pd.DataFrame, column: str = 'Close', span: int = 30) -> pd.Series:
    """Calcula la media móvil exponencial (EMA).

    Args:
        df: DataFrame con datos de precios.
        column: Nombre de la columna sobre la que aplicar la media.
        span: Parámetro `span` de la EMA (equivalente aproximadamente a ventana).

    Returns:
        Serie con la EMA alineada al índice del DataFrame.
    """
    return df[column].ewm(span=span, adjust=False).mean()


def generate_ma_signals(
    df: pd.DataFrame,
    short_window: int = 30,
    long_window: int = 100,
    column: str = 'Close',
    ma_type: str = 'SMA',
) -> pd.DataFrame:
    """Genera medias móviles y señales de cruce corto/largo.

    Señales: 1 (posición larga cuando la MA corta cruza por encima),
    -1 (salida/posición corta cuando cruza por debajo), 0 (neutral).

    Args:
        df: DataFrame con columna de precios.
        short_window: Ventana para la media corta.
        long_window: Ventana para la media larga.
        column: Columna de precios.
        ma_type: 'SMA' o 'EMA'.

    Returns:
        Copia del DataFrame con columnas añadidas: `MA_short`, `MA_long`, `signal`.
    """
    df = df.copy()
    if ma_type.upper() == 'EMA':
        df['MA_short'] = compute_ema(df, column=column, span=short_window)
        df['MA_long'] = compute_ema(df, column=column, span=long_window)
    else:
        df['MA_short'] = compute_sma(df, column=column, window=short_window)
        df['MA_long'] = compute_sma(df, column=column, window=long_window)

    # Señal basada en el cruce de medias (1, 0, -1)
    df['signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'signal'] = -1

    # Opcional: detectar cruces exactos
    df['signal_change'] = df['signal'].diff().fillna(0)
    return df


def example_usage():
    """Ejemplo simple de uso con DataProcessor.

    Descarga datos del IBEX (usando la clase existente), añade indicadores, y
    calcula medias móviles para comparar con el enfoque del Autoencoder.
    """
    processor = DataProcessor()
    df = processor.download_data()
    processor.add_features()

    # Usamos la columna Close para las medias móviles
    df_ma = generate_ma_signals(df, short_window=50, long_window=200, column='Close', ma_type='SMA')

    print(df_ma[['Close', 'MA_short', 'MA_long', 'signal']].tail(10))


if __name__ == '__main__':
    example_usage()
