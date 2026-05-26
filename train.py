"""
train.py — Script principal de entrenamiento del CNN-AE.

Flujo completo:
  1. DataProcessor descarga y procesa los datos (compañero Cristian)
  2. Seleccionamos las 5 features relevantes
  3. Dividimos, escalamos y creamos las secuencias
  4. Entrenamos el ConvAutoencoder solo con datos pre-2020
  5. Calculamos errores de reconstrucción sobre datos de test (2020+)
  6. Calibramos umbrales y visualizamos el "semáforo de riesgo"
"""

import numpy as np
import pandas as pd

from load_data import DataProcessor
from cnn_ae import (
    FEATURE_COLS,
    ConvAutoencoder,
    calibrate_thresholds,
    classify_signals,
    compute_reconstruction_errors,
    load_model,
    plot_semaforo,
    plot_training_history,
    save_model,
    train_autoencoder,
)

# ── PARÁMETROS GLOBALES ────────────────────────────────────────────────────────
TICKER        = "^IBEX"
START_DATE    = "2000-01-01"
END_DATE      = "2024-12-31"
STABILITY_END = "2019-12-31"  # hasta aquí es "mercado normal" para el entrenamiento
WINDOW_SIZE   = 30            # días por ventana (tensor)
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3


def main():
    # ── 1. DESCARGA Y FEATURE ENGINEERING ─────────────────────────────────────
    print("=" * 60)
    print("PASO 1: Descarga de datos y feature engineering")
    print("=" * 60)
    processor = DataProcessor(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)
    processor.download_data()
    processor.add_features()

    # ── 2. SELECCIÓN DE FEATURES ───────────────────────────────────────────────
    # Usamos solo las 5 columnas de FEATURE_COLS en lugar de todas las del DataFrame.
    # Esto reduce el ruido y enfoca el modelo en las señales más relevantes.
    print(f"\nFeatures seleccionadas: {FEATURE_COLS}")
    full_df = processor.data[FEATURE_COLS]

    # ── 3. SPLIT TEMPORAL ──────────────────────────────────────────────────────
    print(f"\nPASO 2: División temporal (train hasta {STABILITY_END})")
    # Nota: usamos split_data del DataProcessor pero sobre nuestro subconjunto
    train_df = full_df[:STABILITY_END]
    test_df  = full_df[STABILITY_END:]
    print(f"  Train: {len(train_df)} días ({train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f"  Test:  {len(test_df)} días ({test_df.index[0].date()} → {test_df.index[-1].date()})")

    # ── 4. ESCALADO (sin data leakage) ─────────────────────────────────────────
    print("\nPASO 3: Escalado con MinMaxScaler (ajustado solo sobre train)")
    # El scaler aprende min/max SOLO del periodo estable → no "ve" los extremos del crash
    train_scaled, test_scaled = processor.scale_data(train_df, test_df)

    # ── 5. CREACIÓN DE SECUENCIAS (tensores 3D) ────────────────────────────────
    print(f"\nPASO 4: Creación de secuencias (ventana de {WINDOW_SIZE} días)")
    X_train = processor.create_sequences(train_scaled, window_size=WINDOW_SIZE)
    X_test  = processor.create_sequences(test_scaled,  window_size=WINDOW_SIZE)
    print(f"  X_train: {X_train.shape}  ← (muestras, {WINDOW_SIZE} días, {len(FEATURE_COLS)} features)")
    print(f"  X_test:  {X_test.shape}")

    # Fechas alineadas con las secuencias de test (el índice i apunta al último día de la ventana)
    test_dates = test_df.index[WINDOW_SIZE:]

    # ── 6. ENTRENAMIENTO DEL CNN-AE ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PASO 5: Entrenamiento del Autoencoder Convolucional")
    print("=" * 60)
    n_features = X_train.shape[2]
    model = ConvAutoencoder(n_features=n_features, seq_len=WINDOW_SIZE)
    print(model)

    history = train_autoencoder(
        model,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )
    plot_training_history(history)
    save_model(model, path="cnn_ae_weights.pth")

    # ── 7. ERRORES DE RECONSTRUCCIÓN ───────────────────────────────────────────
    print("\nPASO 6: Cálculo de errores de reconstrucción")
    # Errores sobre train → para calibrar los umbrales del semáforo
    train_errors = compute_reconstruction_errors(model, X_train)
    # Errores sobre test → para ver si detecta el crash del COVID-19
    test_errors  = compute_reconstruction_errors(model, X_test)
    print(f"  Error medio en train: {train_errors.mean():.6f}")
    print(f"  Error medio en test:  {test_errors.mean():.6f}")

    # ── 8. CALIBRACIÓN DEL SEMÁFORO ────────────────────────────────────────────
    print("\nPASO 7: Calibración del semáforo (umbrales por percentil)")
    thresholds = calibrate_thresholds(train_errors, p_amber=90, p_red=97)
    signals = classify_signals(test_errors, thresholds)

    # Resumen de señales
    from collections import Counter
    counts = Counter(signals)
    print(f"  Señales en test → verde: {counts['verde']}, ámbar: {counts['ámbar']}, rojo: {counts['rojo']}")

    # ── 9. VISUALIZACIÓN ───────────────────────────────────────────────────────
    print("\nPASO 8: Visualización del semáforo de riesgo")
    plot_semaforo(
        dates=test_dates,
        errors=test_errors,
        thresholds=thresholds,
        title=f"Semáforo de Riesgo — {TICKER} (2020–2024)",
    )

    print("\n✓ Pipeline completo finalizado.")
    return model, history, train_errors, test_errors, thresholds, signals


if __name__ == "__main__":
    main()
