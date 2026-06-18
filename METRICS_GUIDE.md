"""
GUÍA DE IMPLEMENTACIÓN: MÉTRICAS DE EVALUACIÓN PARA EL TFM
==========================================================

Este documento explica cómo usar las métricas implementadas para evaluar tu modelo
de Semáforo de Inversión (CNN-AE + TCN) contra benchmarks.

## Archivos generados:
1. metrics.py ................... Funciones core para cálculo de Sharpe y MDD
2. evaluation_pipeline.py ....... Pipeline completo de comparación
3. notebooks/metrics_evaluation.ipynb .. Notebook interactivo con ejemplos

## Flujo de trabajo rápido
=========================

### OPCIÓN A: Uso en Jupyter Notebook (Google Colab)

1. Abre: notebooks/metrics_evaluation.ipynb

2. Ejecuta las celdas en orden:
   - Importa librerías
   - Descarga datos históricos
   - Calcula Sharpe Ratio
   - Calcula Maximum Drawdown
   - Compara estrategias
   - Evalúa criterios TFM

3. Personaliza con datos de tu modelo:
   - Reemplaza 'precios_semaforo' con outputs de tu CNN-AE+TCN
   - Especifica período 2026
   - Compara con IBEX 35 (local) y S&P 500 (internacional)


### OPCIÓN B: Uso programático en Python

from metrics import (
    calculate_strategy_metrics,
    compare_strategies,
    evaluate_tfm_criteria,
)

# Tus datos (precios de cierre)
precios_semaforo = np.array([...])  # De tu modelo CNN-AE+TCN
precios_ma = np.array([...])        # Medias Móviles SMA 50/200
precios_buy_hold = np.array([...])  # Buy & Hold IBEX

# Comparar estrategias
strategies = {
    "Semáforo (CNN-AE+TCN)": precios_semaforo,
    "Medias Móviles (SMA 50/200)": precios_ma,
    "Buy & Hold (IBEX)": precios_buy_hold,
}

df_comparison = compare_strategies(strategies, verbose=True)

# Evaluar criterios TFM
resultado = evaluate_tfm_criteria(
    sharpe_semaforo=0.85,
    sharpe_sp500=0.60,
    mdd_semaforo=-0.25,
    mdd_buy_hold=-0.40,
    retorno_semaforo=0.25,
    retorno_ma_referencia=0.18,
    verbose=True
)


## Explicación de cada métrica
=====================================

### 1. SHARPE RATIO (Eficiencia de la inversión)

¿Qué mide?
  - Cuánto retorno GANASTE por cada unidad de RIESGO que asumiste
  - Fórmula: Sharpe = (E[Retorno] - Rf) / σ[Retorno] × √252

Interpretación:
  - Sharpe > 1.0: Excelente (ganas mucho por poco riesgo)
  - Sharpe > 0.5: Bueno (retorno > volatilidad)
  - Sharpe = 0: Retorno igual a tasa libre de riesgo
  - Sharpe < 0: Peor que dejar dinero en depósito

Criterio TFM:
  ✓ Sharpe_Semáforo - Sharpe_SP500 ≥ 0.25
  → Tu modelo debe ser 0.25 puntos superior al S&P 500

Ejemplo:
  - Si Sharpe_SP500 = 0.60
  - Entonces Sharpe_Semáforo ≥ 0.85


### 2. MAXIMUM DRAWDOWN (Preservación de capital)

¿Qué mide?
  - La PEOR CAÍDA desde tu pico máximo en todo el período
  - Fórmula: MDD = (Valor_mínimo - Pico_anterior) / Pico_anterior
  - Siempre es NEGATIVO (es una pérdida)

Interpretación:
  - MDD = -10%: En el peor momento, perdiste 10%
  - MDD = -50%: En el peor momento, perdiste 50% (crisis grave)
  - MDD cercano a 0: Muy estable (mercado muy positivo)

Criterio TFM:
  ✓ MDD_Semáforo debe ser ≥ 15% MEJOR que MDD_Buy&Hold
  → Si Buy&Hold sufre -40%, Semáforo debe tener MDD ≥ -25%

Fórmula: MDD_BuyHold - MDD_Semáforo ≥ 0.15

Ejemplo:
  - MDD_Buy&Hold = -0.40 (-40%)
  - MDD_Semáforo debe ser ≤ -0.25 (-25%)
  - Mejora = -0.40 - (-0.25) = 0.15 ✓ CUMPLE


### 3. RETORNO NETO / VOLATILIDAD ANUALIZADA

Retorno Neto:
  - Ganancia total en todo el período
  - Fórmula: (Precio_Final - Precio_Inicial) / Precio_Inicial

Volatilidad:
  - Medida de variabilidad de retornos
  - Fórmula: σ[retornos_diarios] × √252
  - Valores altos = mercado errático
  - Valores bajos = mercado predecible


## GUÍA DETALLADA POR ESTRATEGIA
=====================================

### ESTRATEGIA 1: Semáforo (CNN-AE + TCN) - TU MODELO

¿Cómo generar precios?

from metrics import simulate_semaforo_strategy
from load_data import DataProcessor

# Obtener errores de reconstrucción de tu modelo
processor = DataProcessor()
df = processor.download_data()
processor.add_features()

# Entrenar CNN-AE, obtener MSE
model = ConvAutoencoder(...)
errors_mse = compute_reconstruction_errors(model, X_test)

# Generar precios de la estrategia
precios_semaforo = simulate_semaforo_strategy(
    df=df_test,
    reconstruction_errors=errors_mse,
    threshold_green=0.5,  # MSE bajo → compra
    threshold_red=1.5,    # MSE alto → venta
)

# Calcular métricas
metrics = calculate_strategy_metrics(precios_semaforo, "Semáforo CNN-AE+TCN")


### ESTRATEGIA 2: Medias Móviles (SMA 50/200) - REFERENCIA TÉCNICA

¿Cómo generar precios?

from medias_moviles import generate_ma_signals
from evaluation_pipeline import generate_ma_strategy_prices

precios_ma = generate_ma_strategy_prices(
    df=df_test,
    short_window=50,
    long_window=200,
    column='Close'
)

# Calcular métricas
metrics = calculate_strategy_metrics(precios_ma, "SMA 50/200")

Lógica:
  - Si SMA50 > SMA200: COMPRA (posición larga +1)
  - Si SMA50 < SMA200: VENTA (posición corta -1)


### ESTRATEGIA 3: Buy & Hold - BENCHMARK PASIVO

¿Cómo generar precios?

# Simple: son los precios sin modificar del índice
precios_buy_hold = df['Close'].values

# Calcular métricas
metrics = calculate_strategy_metrics(precios_buy_hold, "Buy & Hold IBEX")


### ESTRATEGIA 4 (OPCIONAL): S&P 500 - BENCHMARK INTERNACIONAL

Para el Criterio 3 (Sharpe), necesitas benchmarkear contra S&P 500:

import yfinance as yf

sp500 = yf.download('^GSPC', start='2026-01-01', end='2026-06-30')
precios_sp500 = sp500['Close'].values

metrics_sp500 = calculate_strategy_metrics(precios_sp500, "S&P 500")
sharpe_sp500 = metrics_sp500['sharpe_ratio']


## CRITERIOS DE ÉXITO DEL TFM
=====================================

Debes cumplir TODOS ESTOS 3 criterios:

┌─────────────────────────────────────────────────────────────────┐
│ CRITERIO 1: Rendimiento Financiero Relativo                     │
├─────────────────────────────────────────────────────────────────┤
│ Semáforo debe superar MA 50/200 en al menos 10% de rendimiento  │
│                                                                  │
│ Fórmula:                                                         │
│   Retorno_Semáforo / Retorno_MA ≥ 1.10                          │
│                                                                  │
│ Ejemplo:                                                         │
│   - Si MA_50/200 retorna +18%                                   │
│   - Entonces Semáforo debe retornar ≥ +19.8% (+18% × 1.10)     │
│                                                                  │
│ Código:                                                          │
│   retorno_rel = retorno_semaforo / retorno_ma                  │
│   criterio1 = retorno_rel >= 1.10                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CRITERIO 2: Preservación de Capital                             │
├─────────────────────────────────────────────────────────────────┤
│ Semáforo debe tener MDD ≥ 15% mejor que Buy & Hold              │
│                                                                  │
│ Fórmula:                                                         │
│   MDD_BuyHold - MDD_Semáforo ≥ 0.15                             │
│                                                                  │
│ Ejemplo:                                                         │
│   - Buy & Hold sufre MDD de -40%                                │
│   - Semáforo debe tener MDD ≤ -25% (mejora de 15%)             │
│   - Mejora: -40% - (-25%) = 15% ✓                              │
│                                                                  │
│ Código:                                                          │
│   mejora = mdd_buy_hold - mdd_semaforo                          │
│   criterio2 = mejora >= 0.15                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CRITERIO 3: Calidad de Inversión (Eficiencia)                   │
├─────────────────────────────────────────────────────────────────┤
│ Sharpe_Semáforo debe superar Sharpe_SP500 en al menos 0.25      │
│                                                                  │
│ Fórmula:                                                         │
│   Sharpe_Semáforo - Sharpe_SP500 ≥ 0.25                         │
│                                                                  │
│ Ejemplo:                                                         │
│   - S&P 500 tiene Sharpe = 0.60                                 │
│   - Semáforo debe tener Sharpe ≥ 0.85                           │
│   - Exceso: 0.85 - 0.60 = 0.25 ✓                               │
│                                                                  │
│ Código:                                                          │
│   exceso_sharpe = sharpe_semaforo - sharpe_sp500               │
│   criterio3 = exceso_sharpe >= 0.25                             │
└─────────────────────────────────────────────────────────────────┘


## EJERCICIO PRÁCTICO: CALCULAR LAS 3 MÉTRICAS
=====================================

Datos simulados (reemplaza con tus datos reales):

import numpy as np
from metrics import calculate_strategy_metrics, evaluate_tfm_criteria

# 1. Generar precios simulados de ejemplo
n_dias = 252  # 1 año
precio_base = np.linspace(100, 120, n_dias)

# Semáforo (mejor en crisis)
ruido_semaforo = np.random.normal(0, 0.5, n_dias)
precios_semaforo = precio_base + np.cumsum(ruido_semaforo)

# MA 50/200 (referencia)
ruido_ma = np.random.normal(0, 0.8, n_dias)
precios_ma = precio_base + np.cumsum(ruido_ma)

# Buy & Hold (pasivo)
ruido_bh = np.random.normal(0, 1.0, n_dias)
precios_bh = precio_base + np.cumsum(ruido_bh)

# 2. Calcular métricas
metrics_semaforo = calculate_strategy_metrics(precios_semaforo, "Semáforo")
metrics_ma = calculate_strategy_metrics(precios_ma, "MA 50/200")
metrics_bh = calculate_strategy_metrics(precios_bh, "Buy & Hold")

# 3. Extraer valores clave
sharpe_semaforo = metrics_semaforo['sharpe_ratio']
sharpe_sp500 = 0.60  # Datos reales
mdd_semaforo = metrics_semaforo['max_drawdown']
mdd_bh = metrics_bh['max_drawdown']
ret_semaforo = metrics_semaforo['cumulative_return']
ret_ma = metrics_ma['cumulative_return']

# 4. Evaluar criterios
resultado = evaluate_tfm_criteria(
    sharpe_semaforo=sharpe_semaforo,
    sharpe_sp500=sharpe_sp500,
    mdd_semaforo=mdd_semaforo,
    mdd_buy_hold=mdd_bh,
    retorno_semaforo=ret_semaforo,
    retorno_ma_referencia=ret_ma,
    verbose=True
)

# Imprimir resultado
if resultado['all_criteria_passed']:
    print("✓ TFM CUMPLE TODOS LOS CRITERIOS")
else:
    print("✗ TFM NO CUMPLE TODOS LOS CRITERIOS")


## INTEGRACIÓN CON TU PIPELINE
=====================================

Paso 1: Entrenar CNN-AE
→ train.py genera X_train, X_test, modelo entrenado

Paso 2: Obtener errores de reconstrucción
→ compute_reconstruction_errors(modelo, X_test)

Paso 3: Generar señales del Semáforo
→ Desde MSE: Verde/Ámbar/Rojo

Paso 4: Calcular precios de la estrategia
→ simulate_semaforo_strategy(df, mse_errors)

Paso 5: Calcular métricas
→ calculate_strategy_metrics(precios_semaforo)

Paso 6: Comparar con benchmarks
→ compare_strategies({...})

Paso 7: Evaluar criterios TFM
→ evaluate_tfm_criteria(...)

Paso 8: Documentar para el tribunal
→ Tabla resumen + gráficos


## REFERENCIAS MATEMÁTICAS
=====================================

SHARPE RATIO ANUALIZADO:
  Sharpe = [(μ_retornos - R_f) / σ_retornos] × √252
  
  Donde:
    μ_retornos = promedio de retornos diarios
    R_f = tasa libre de riesgo diaria (0.02/252)
    σ_retornos = desviación estándar de retornos
    252 = días de trading por año

MAXIMUM DRAWDOWN:
  Paso 1: Calcular el máximo acumulado hasta cada día
    running_max[i] = max(precios[0:i])
  
  Paso 2: Calcular drawdown en cada punto
    drawdown[i] = (precios[i] - running_max[i]) / running_max[i]
  
  Paso 3: El MDD es el drawdown mínimo
    MDD = min(drawdown)

RETORNO NETO:
  Return = (P_final - P_inicial) / P_inicial

VOLATILIDAD ANUALIZADA:
  Vol = σ[retornos_diarios] × √252


## TROUBLESHOOTING
=====================================

Problema: El Sharpe Ratio es negativo
Solución: La estrategia tiene retornos menores que la tasa libre de riesgo.
         Revisa tus cálculos de entrada y parámetros de entrenamiento.

Problema: El MDD es 0 o muy pequeño
Solución: Tu estrategia no ha sufrido caídas (poco realista).
         Verifica que los datos incluyan periodos de crisis (2020).

Problema: La comparación no funciona
Solución: Asegúrate de que todos los arrays tienen la misma longitud.
         Alinea las fechas de los precios.

Problema: ValueError: shapes do not match
Solución: Las longitudes de precios no coinciden entre estrategias.
         Reindexea todas al mismo período temporal.


## ARCHIVOS RECOMENDADOS PARA TU TESIS
=====================================

1. metrics.py - Código fuente
   - Copiar como APÉNDICE A

2. evaluation_pipeline.py - Script de evaluación
   - Usar como referencia metodológica

3. notebooks/metrics_evaluation.ipynb - Notebook ejecutable
   - Demostración práctica

4. Tabla resumen de métricas (CSV/DataFrame)
   - Incluir en sección de resultados

5. Gráficos comparativos (PNG)
   - Diagramas de barras y series temporales


## CONTACTO Y SOPORTE
=====================================

Para dudas sobre:
- Sharpe Ratio: https://en.wikipedia.org/wiki/Sharpe_ratio
- Maximum Drawdown: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp
- CNN-AE: Ver cnn_ae.py
- TCN: Ver tcn_model.py

Autor: Carlos Seguí Martínez
Proyecto: UNIR MIA - TFM 2026
"""

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
