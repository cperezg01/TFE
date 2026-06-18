"""
metrics.py — Cálculo de métricas de rendimiento financiero.

Proporciona funciones para evaluar estrategias de inversión:
- Ratio de Sharpe (rendimiento ajustado por riesgo)
- Maximum Drawdown (pérdida máxima desde pico)
- Retorno neto y volatilidad

Uso:
    - Comparar CNN-AE + TCN vs Medias Móviles vs Buy & Hold
    - Evaluar criterios de éxito del TFM
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def compute_returns(prices: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Calcula los retornos logarítmicos diarios a partir de precios.
    
    Args:
        prices: Array o Serie de precios (cierre diario).
    
    Returns:
        Array de retornos logarítmicos.
    
    Ejemplo:
        >>> prices = np.array([100, 102, 101, 103])
        >>> returns = compute_returns(prices)
        >>> returns  # [log(102/100), log(101/102), log(103/101)]
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    return np.diff(np.log(prices))


def compute_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calcula el Ratio de Sharpe anualizado.
    
    Fórmula:
        Sharpe Ratio = (E[retorno] - rf) / σ[retorno] * sqrt(252)
    
    Args:
        returns: Array de retornos diarios (en formato decimal, ej: 0.01 = 1%).
        risk_free_rate: Tasa libre de riesgo anual (default 2%).
        periods_per_year: Días de trading por año (default 252).
    
    Returns:
        Ratio de Sharpe anualizado (float).
    
    Criterio de éxito del TFM:
        Sharpe_modelo - Sharpe_SP500 >= 0.25
    
    Ejemplo:
        >>> diarios = np.array([0.001, -0.002, 0.0015, 0.001])
        >>> sharpe = compute_sharpe_ratio(diarios, risk_free_rate=0.02)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Retorno diario libre de riesgo
    daily_rf = risk_free_rate / periods_per_year
    
    # Exceso de retorno medio diario
    mean_excess_return = np.mean(returns) - daily_rf
    
    # Volatilidad diaria
    std_return = np.std(returns, ddof=1)
    
    # Evitar división por cero
    if std_return == 0:
        return 0.0
    
    # Ratio de Sharpe diario, anualizado
    sharpe_daily = mean_excess_return / std_return
    sharpe_annual = sharpe_daily * np.sqrt(periods_per_year)
    
    return sharpe_annual


def compute_maximum_drawdown(prices: Union[np.ndarray, pd.Series]) -> Tuple[float, int, int]:
    """
    Calcula el Maximum Drawdown (MDD) y las fechas asociadas.
    
    El MDD es la pérdida máxima desde un pico anterior:
        MDD = (valor_mínimo - pico_anterior) / pico_anterior
    
    Criterio de éxito del TFM:
        MDD_semaforo <= MDD_buy_hold - 0.15  (al menos 15% menor)
    
    Args:
        prices: Array o Serie de precios de cierre.
    
    Returns:
        Tupla (mdd, peak_idx, trough_idx):
            - mdd: Maximum Drawdown (valor negativo, ej: -0.35 = -35%).
            - peak_idx: Índice del pico anterior.
            - trough_idx: Índice del valle.
    
    Ejemplo:
        >>> prices = np.array([100, 110, 105, 95, 100, 90])
        >>> mdd, peak_idx, trough_idx = compute_maximum_drawdown(prices)
        >>> mdd  # -0.1818... (de 110 a 90 = -18.18%)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    prices = np.asarray(prices, dtype=float)
    
    if len(prices) < 2:
        return 0.0, 0, 0
    
    # Calcula el máximo acumulado hasta cada punto (running maximum)
    running_max = np.maximum.accumulate(prices)
    
    # Drawdown en cada punto: (precio - máximo anterior) / máximo anterior
    drawdown = (prices - running_max) / running_max
    
    # Encuentra el MDD (máxima pérdida)
    mdd_idx = np.argmin(drawdown)
    mdd = drawdown[mdd_idx]
    
    # Encuentra el pico anterior más cercano
    peak_idx = np.argmax(prices[:mdd_idx + 1])
    
    return mdd, peak_idx, mdd_idx


def compute_cumulative_return(prices: Union[np.ndarray, pd.Series]) -> float:
    """
    Calcula el retorno neto acumulado desde el inicio hasta el final.
    
    Fórmula:
        Retorno = (precio_final - precio_inicial) / precio_inicial
    
    Args:
        prices: Array o Serie de precios.
    
    Returns:
        Retorno neto (ej: 0.25 = +25%, -0.15 = -15%).
    
    Ejemplo:
        >>> prices = np.array([100, 110, 105, 115])
        >>> ret = compute_cumulative_return(prices)
        >>> ret  # 0.15 (+15%)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    prices = np.asarray(prices, dtype=float)
    
    if len(prices) < 2 or prices[0] == 0:
        return 0.0
    
    return (prices[-1] - prices[0]) / prices[0]


def compute_volatility(
    returns: Union[np.ndarray, pd.Series],
    periods_per_year: int = 252,
) -> float:
    """
    Calcula la volatilidad anualizada a partir de retornos diarios.
    
    Fórmula:
        Volatilidad_anual = σ[retorno_diario] * sqrt(252)
    
    Args:
        returns: Array de retornos diarios.
        periods_per_year: Días de trading por año.
    
    Returns:
        Volatilidad anualizada (ej: 0.20 = 20%).
    
    Ejemplo:
        >>> retornos = np.array([0.001, -0.002, 0.0015])
        >>> vol = compute_volatility(retornos)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    std_daily = np.std(returns, ddof=1)
    volatility_annual = std_daily * np.sqrt(periods_per_year)
    
    return volatility_annual


def calculate_strategy_metrics(
    prices: Union[np.ndarray, pd.Series],
    strategy_name: str = "Strategy",
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calcula un conjunto completo de métricas para una estrategia.
    
    Retorna:
        - Retorno neto
        - Volatilidad anualizada
        - Ratio de Sharpe
        - Maximum Drawdown
        - Retornos diarios
    
    Args:
        prices: Array o Serie de precios de cierre.
        strategy_name: Nombre de la estrategia (para logging).
        risk_free_rate: Tasa libre de riesgo anual.
    
    Returns:
        Diccionario con todas las métricas.
    
    Ejemplo:
        >>> prices = np.array([100, 102, 101, 103, 102, 105])
        >>> metrics = calculate_strategy_metrics(prices, "Mi Estrategia")
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.4f}")
        >>> print(f"MDD: {metrics['max_drawdown']:.2%}")
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calcula retornos
    returns = compute_returns(prices)
    
    # Calcula métricas
    cumulative_ret = compute_cumulative_return(prices)
    volatility = compute_volatility(returns)
    sharpe = compute_sharpe_ratio(returns, risk_free_rate=risk_free_rate)
    mdd, peak_idx, trough_idx = compute_maximum_drawdown(prices)
    
    metrics = {
        'strategy_name': strategy_name,
        'cumulative_return': cumulative_ret,
        'daily_returns': returns,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'peak_idx': peak_idx,
        'trough_idx': trough_idx,
    }
    
    return metrics


def compare_strategies(
    strategies: dict,
    risk_free_rate: float = 0.02,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compara múltiples estrategias usando métricas financieras.
    
    Entrada esperada:
        strategies = {
            "Semáforo (CNN-AE+TCN)": prices_semaforo,
            "Medias Móviles (SMA 50/200)": prices_ma,
            "Buy & Hold": prices_buy_hold,
        }
    
    Args:
        strategies: Dict {nombre_estrategia: array_de_precios}
        risk_free_rate: Tasa libre de riesgo.
        verbose: Si True, imprime comparación formateada.
    
    Returns:
        DataFrame con métricas de todas las estrategias.
    
    Ejemplo:
        >>> strategies = {
        ...     "CNN-AE": np.array([100, 102, 104, 106]),
        ...     "MA": np.array([100, 101, 103, 105]),
        ... }
        >>> df = compare_strategies(strategies)
        >>> print(df)
    """
    results = []
    
    for name, prices in strategies.items():
        metrics = calculate_strategy_metrics(prices, name, risk_free_rate)
        results.append({
            'Estrategia': name,
            'Retorno Neto': f"{metrics['cumulative_return']:.2%}",
            'Volatilidad': f"{metrics['volatility']:.2%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
        })
    
    df_comparison = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "=" * 70)
        print("COMPARACIÓN DE ESTRATEGIAS")
        print("=" * 70)
        print(df_comparison.to_string(index=False))
        print("=" * 70)
    
    return df_comparison


# ── FUNCIONES AUXILIARES PARA COMPARAR CRITERIOS DE ÉXITO ────────────────────

def evaluate_tfm_criteria(
    sharpe_semaforo: float,
    sharpe_sp500: float,
    mdd_semaforo: float,
    mdd_buy_hold: float,
    retorno_semaforo: float,
    retorno_ma_referencia: float,
    verbose: bool = True,
) -> dict:
    """
    Evalúa si el modelo cumple los 3 criterios de éxito del TFM.
    
    Criterios:
        1. Rendimiento Financiero Relativo: Superar MA en al menos 10% neto.
        2. Preservación de Capital: MDD del Semáforo >= 15% menor que Buy & Hold.
        3. Calidad de Inversión (Eficiencia): Sharpe_Semáforo - Sharpe_SP500 >= 0.25
    
    Args:
        sharpe_semaforo: Sharpe Ratio del modelo híbrido.
        sharpe_sp500: Sharpe Ratio del benchmark S&P 500.
        mdd_semaforo: Maximum Drawdown del Semáforo.
        mdd_buy_hold: Maximum Drawdown del Buy & Hold.
        retorno_semaforo: Retorno neto del Semáforo.
        retorno_ma_referencia: Retorno neto de Medias Móviles.
        verbose: Si True, imprime evaluación.
    
    Returns:
        Dict con evaluación de cada criterio.
    
    Ejemplo:
        >>> result = evaluate_tfm_criteria(
        ...     sharpe_semaforo=1.2, sharpe_sp500=0.8,
        ...     mdd_semaforo=-0.25, mdd_buy_hold=-0.40,
        ...     retorno_semaforo=0.30, retorno_ma_referencia=0.20
        ... )
    """
    # Criterio 1: Rendimiento Financiero Relativo (10% mejor)
    retorno_relativo = retorno_semaforo / retorno_ma_referencia if retorno_ma_referencia != 0 else 1.0
    criterio1_passed = retorno_relativo >= 1.10
    criterio1_exceso = (retorno_relativo - 1.0) * 100
    
    # Criterio 2: Preservación de Capital (15% menos drawdown)
    mdd_mejora = mdd_buy_hold - mdd_semaforo  # Valor positivo = mejora
    criterio2_passed = mdd_mejora >= 0.15
    
    # Criterio 3: Calidad de Inversión (Sharpe 0.25 superior)
    sharpe_exceso = sharpe_semaforo - sharpe_sp500
    criterio3_passed = sharpe_exceso >= 0.25
    
    result = {
        'criterio_1_rendimiento': {
            'passed': criterio1_passed,
            'valor': retorno_relativo,
            'exceso_pct': criterio1_exceso,
            'requerido': 0.10,  # 10% de exceso
        },
        'criterio_2_drawdown': {
            'passed': criterio2_passed,
            'mejora': mdd_mejora,
            'requerido': 0.15,  # 15% de mejora
        },
        'criterio_3_sharpe': {
            'passed': criterio3_passed,
            'exceso': sharpe_exceso,
            'requerido': 0.25,
        },
        'all_criteria_passed': criterio1_passed and criterio2_passed and criterio3_passed,
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUACIÓN DE CRITERIOS DE ÉXITO DEL TFM")
        print("=" * 70)
        
        print(f"\n✓ Criterio 1: Rendimiento Financiero Relativo")
        print(f"  Semáforo retorno: {retorno_semaforo:.2%}")
        print(f"  MA referencia: {retorno_ma_referencia:.2%}")
        print(f"  Relativo: {retorno_relativo:.2%} (requerido ≥ 110%)")
        print(f"  Exceso: {criterio1_exceso:.1f}% {'✓ CUMPLE' if criterio1_passed else '✗ NO CUMPLE'}")
        
        print(f"\n✓ Criterio 2: Preservación de Capital")
        print(f"  Semáforo MDD: {mdd_semaforo:.2%}")
        print(f"  Buy & Hold MDD: {mdd_buy_hold:.2%}")
        print(f"  Mejora: {mdd_mejora:.2%} (requerido ≥ 15%)")
        print(f"  Estado: {'✓ CUMPLE' if criterio2_passed else '✗ NO CUMPLE'}")
        
        print(f"\n✓ Criterio 3: Calidad de Inversión (Eficiencia)")
        print(f"  Sharpe Semáforo: {sharpe_semaforo:.4f}")
        print(f"  Sharpe S&P 500: {sharpe_sp500:.4f}")
        print(f"  Exceso: {sharpe_exceso:.4f} (requerido ≥ 0.25)")
        print(f"  Estado: {'✓ CUMPLE' if criterio3_passed else '✗ NO CUMPLE'}")
        
        print(f"\n{'='*70}")
        if result['all_criteria_passed']:
            print("✓✓✓ TODOS LOS CRITERIOS CUMPLEN ✓✓✓")
        else:
            print("✗✗✗ ALGUNOS CRITERIOS NO SE CUMPLEN ✗✗✗")
        print(f"{'='*70}\n")
    
    return result


# ── EJEMPLO DE USO ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulación de datos para demostración
    np.random.seed(42)
    
    # Precios de ejemplo (3 estrategias simuladas)
    n_dias = 252  # 1 año de trading
    
    # Precio base
    precio_base = np.linspace(100, 115, n_dias)
    
    # Estrategia 1: Semáforo (CNN-AE+TCN) — mejor en crisis
    ruido_semaforo = np.random.normal(0, 0.8, n_dias - 1)
    retornos_semaforo = np.concatenate([[0], ruido_semaforo]) / 100
    precio_semaforo = precio_base * np.exp(np.cumsum(retornos_semaforo))
    
    # Estrategia 2: Medias Móviles (referencia)
    ruido_ma = np.random.normal(0, 1.0, n_dias - 1)
    retornos_ma = np.concatenate([[0], ruido_ma]) / 100
    precio_ma = precio_base * np.exp(np.cumsum(retornos_ma))
    
    # Estrategia 3: Buy & Hold (pasivo)
    ruido_bh = np.random.normal(0, 1.2, n_dias - 1)
    retornos_bh = np.concatenate([[0], ruido_bh]) / 100
    precio_bh = precio_base * np.exp(np.cumsum(retornos_bh))
    
    # Comparación de estrategias
    strategies = {
        "Semáforo (CNN-AE+TCN)": precio_semaforo,
        "Medias Móviles (SMA 50/200)": precio_ma,
        "Buy & Hold": precio_bh,
    }
    
    df_comparison = compare_strategies(strategies, verbose=True)
    
    # Obtener métricas individuales
    metrics_semaforo = calculate_strategy_metrics(precio_semaforo, "Semáforo")
    metrics_sp500 = calculate_strategy_metrics(precio_bh, "S&P 500")  # simulado
    metrics_ma = calculate_strategy_metrics(precio_ma, "Medias Móviles")
    
    # Evaluar criterios de TFM
    evaluate_tfm_criteria(
        sharpe_semaforo=metrics_semaforo['sharpe_ratio'],
        sharpe_sp500=metrics_sp500['sharpe_ratio'],
        mdd_semaforo=metrics_semaforo['max_drawdown'],
        mdd_buy_hold=metrics_bh['max_drawdown'],
        retorno_semaforo=metrics_semaforo['cumulative_return'],
        retorno_ma_referencia=metrics_ma['cumulative_return'],
        verbose=True,
    )
