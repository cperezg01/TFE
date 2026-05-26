# TFM — Detección de cambios de mercado con Deep Learning no supervisado

## Contexto rápido
Proyecto grupal UNIR (MIA). Sistema "semáforo de riesgo" para inversores que detecta cambios de régimen en mercados financieros usando CNN-AE + TCN.

## Reparto de trabajo
| Alumno | Parte | Archivo principal |
|--------|-------|-------------------|
| Cristian José Pérez García | Pipeline de datos | `load_data.py` |
| **Carlos Seguí Martínez** | **CNN-AE (nuestra parte)** | **`cnn_ae.py`** |
| Alexander Jestaedt Maldonado | TCN | `tcn_model.py` |

## Arquitectura del sistema
1. **DataProcessor** (`load_data.py`) → ya implementado. Descarga IBEX/S&P500, calcula features (Log_Ret, Volatilidad, RSI, MACD, SMA), divide train/test, escala y genera tensores 3D `(muestras, 30_dias, n_features)`.
2. **CNN-AE** (`cnn_ae.py`) → nuestro trabajo. Autoencoder convolucional que aprende la "normalidad" (datos pre-2020) y detecta anomalías por error de reconstrucción (MSE).
3. **TCN** → parte de Alexander. Captura dependencias temporales de largo plazo.
4. **Semáforo** → lógica de umbral sobre el MSE: verde / ámbar / rojo.

## Principio clave del CNN-AE
- Entrenado SOLO con datos de estabilidad (hasta 2019-12-31).
- En inferencia: MSE bajo = mercado normal; MSE alto = anomalía/crisis.
- No predice precios — detecta si el mercado "se parece" a lo que aprendió.

## Instrucciones para el asistente
- **Siempre explicar** qué se hace y por qué antes de escribir código. El usuario quiere entender para defender el trabajo ante el tribunal.
- Explicar las decisiones de arquitectura (por qué Conv1D y no Dense, por qué ese número de filtros, etc.).
- El usuario trabaja en rama `feat/cnn_csm`.
- Framework preferido: Keras/TensorFlow (confirmar si cambia).
- Los notebooks van a ejecutarse en Google Colab con GPU.

## Features usadas (del DataProcessor)
El DataFrame de `load_data.py` tiene: Open, High, Low, Close, Volume, Log_Ret, Volatilidad, RSI_14, MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9, SMA_50, SMA_200.
Para el CNN-AE usamos un subconjunto de 5-6 features relevantes (a definir).

## Estructura de archivos objetivo
```
TFE/
├── load_data.py          # DataProcessor — datos y features (Cristian) ✅
├── cnn_ae.py             # CNN Autoencoder (Carlos) 🔧
├── tcn_model.py          # TCN (Alexander)
├── semaforo.py           # Lógica de detección y visualización
├── train.py              # Script principal de entrenamiento
├── notebooks/
│   └── eda_notebook.ipynb
└── CLAUDE.md             # Este archivo
```
