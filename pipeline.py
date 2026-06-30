"""
pipeline.py — Orquestador completo del TFM (multi-activo)
=========================================================
Ejecuta, EN EL MISMO PIPELINE y de forma secuencial (primero uno, luego otro y
finalmente el último), las cuatro etapas del proyecto sobre TRES activos:

      IBEX 35   ->   Oro (Gold)   ->   S&P 500

Para CADA activo se corren, en orden:

  [1] Descarga e ingeniería de datos        -> load_data.DataProcessor
  [2] Estrategia de Medias Móviles          -> medias_moviles.generate_ma_signals
  [3] Semáforo CNN-AE + TCN
        3a. Entrena el CNN-AE y guarda cnn_ae_weights_<activo>.pth   -> cnn_ae
        3b. Carga ese .pth y entrena el agente TCN                   -> tcn
  [4] Métricas y comparación económica      -> metrics

Al terminar los tres, se imprime un RESUMEN COMPARATIVO entre activos.

USO (desde VS Code):
    Coloca este archivo en la MISMA carpeta que load_data.py, medias_moviles.py,
    cnn_ae.py, tcn.py y metrics.py, y ejecútalo:

        python pipeline.py

NOTA sobre tcn.py:
    tcn.py no se puede ejecutar de forma autónoma tal cual (le falta la
    constante SEMAFORO_COLORS, su run_inference no acepta 'coverage' y nunca
    genera la columna 'r_pred'). Este pipeline NO usa el main() de tcn: importa
    solo sus piezas limpias (modelos y helpers de entrenamiento) y aporta sus
    propias versiones corregidas de run_inference y apply_trend_gate.
"""

from __future__ import annotations
import os

# ── Evitar el crash del kernel por OpenMP/MKL duplicado (torch + numpy) ────────
# Estas variables DEBEN definirse ANTES de importar numpy/torch. Si el kernel de
# Jupyter se cae sin traceback, casi siempre es por la librería libiomp5
# duplicada; esto lo neutraliza.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Evita el ACCESS_VIOLATION (ExitCode 0xC0000005) por choque de MKL torch<->numpy:
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import gc
import re
import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from load_data import DataProcessor
from medias_moviles import generate_ma_signals
from cnn_ae import (
    ConvAutoencoder, train_autoencoder, save_model, plot_training_history,
)
from tcn import (
    load_cnn_ae, SemaforoAgent,
    build_zf_sequences, forward_returns, make_labels, train_supervised,
    evaluate_signals, regression_metrics, print_metrics,
)
from metrics import (
    calculate_strategy_metrics, compare_strategies, evaluate_tfm_criteria,
)

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVOS A EVALUAR  (se procesan en este orden, uno tras otro)
# ══════════════════════════════════════════════════════════════════════════════
ASSETS = [
    {"ticker": "^IBEX", "name": "IBEX 35"},
    {"ticker": "GC=F",  "name": "Oro (Gold)"},   # futuros del oro COMEX; alt.: "GLD"
    {"ticker": "^GSPC", "name": "S&P 500"},
]

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES (comunes a todos los activos)
# ══════════════════════════════════════════════════════════════════════════════
START_DATE       = "2000-01-01"
END_DATE         = "2024-12-31"
STABILITY_END    = "2019-12-31"      # el CNN-AE solo "ve" el periodo estable
WINDOW_SIZE      = 30
FEATURE_COLS     = ["Log_Ret", "Volatilidad", "RSI_14", "MACD_12_26_9", "SMA_50"]

# CNN-AE
AE_EPOCHS        = 50
FORCE_RETRAIN_AE = False              # True -> reentrena aunque exista el .pth

# Agente TCN (multitarea: clasificación direccional + regresión de retorno)
TCN_SEQ_LEN      = 10
TCN_CHANNELS     = [32, 16]
TCN_KERNEL       = 3
TCN_DROPOUT      = 0.3
AGENT_EPOCHS     = 300
LR               = 3e-4
WEIGHT_DECAY     = 1e-3
LABEL_SMOOTH     = 0.05
LAMBDA_REG       = 10.0
HORIZON          = 20                 # días vista para la etiqueta direccional
COVERAGE         = 0.30              # fracción de días con señal (resto HOLD)

# Filtro de tendencia (gate)
GATE_FAST        = 50
GATE_DD_LOOK     = 20
GATE_DD_THR      = -0.08

# Medias móviles
MA_SHORT         = 50
MA_LONG          = 200

# Backtest / métricas
ALLOW_SHORT      = False             # largo/efectivo (mejor para preservar capital)
RISK_FREE        = 0.02
SHOW_PLOTS       = True              # pon False para ejecución desatendida

# Origen de descarga: "requests" (directo a Yahoo, sin curl_cffi) o "yfinance".
# Por defecto "requests" porque el curl_cffi de yfinance provoca segfaults en
# algunos Windows/Anaconda. Cambia a "yfinance" si arreglas curl_cffi.
DOWNLOADER       = "requests"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
SEMAFORO_COLORS = {0: ("#f59e0b", "HOLD"), 1: ("#22c55e", "BUY"), 2: ("#ef4444", "SELL")}


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════════════════
def safe_token(ticker: str) -> str:
    """Convierte un ticker (p.ej. '^GSPC', 'GC=F') en un token de archivo seguro."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")


def download_yahoo_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios diarios de Yahoo Finance con 'requests' (API v8 chart),
    sin pasar por yfinance/curl_cffi. Devuelve un DataFrame con columnas
    Open/High/Low/Close/Volume e índice de fechas. 'Close' es el cierre ajustado
    (equivalente a auto_adjust=True).
    """
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": p1, "period2": p2, "interval": "1d",
              "events": "div,splits", "includeAdjustedClose": "true"}
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0 Safari/537.36")}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo devolvió error para {ticker}: {chart['error']}")
    result = chart.get("result")
    if not result or "timestamp" not in result[0]:
        raise RuntimeError(f"Yahoo no devolvió datos para {ticker} "
                           f"(¿ticker o rango de fechas incorrectos?).")
    res = result[0]
    ts = res["timestamp"]
    quote = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose", [{}])[0].get("adjclose"))
    idx = pd.to_datetime(ts, unit="s").normalize()
    df = pd.DataFrame({
        "Open":   quote.get("open"),
        "High":   quote.get("high"),
        "Low":    quote.get("low"),
        "Close":  adj if adj is not None else quote.get("close"),
        "Volume": quote.get("volume"),
    }, index=idx)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")]
    df.dropna(subset=["Close"], inplace=True)
    return df


def make_windows(d, ws=WINDOW_SIZE):
    """Convierte una serie 2D (T, n_feat) en ventanas (T-ws, ws, n_feat)."""
    return np.array([d[i:i + ws] for i in range(len(d) - ws)])


def _log_ret(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).fillna(0.0)


def equity_from_positions(close: pd.Series, position: pd.Series) -> pd.Series:
    """
    Construye la curva de capital de una estrategia.
    'position' (en {-1, 0, 1}) es la posición con la que se ENTRA cada día; el
    retorno se materializa al día siguiente (position.shift(1)) para evitar
    look-ahead. La curva arranca en el primer precio del periodo para que todas
    las estrategias sean comparables.
    """
    position = position.reindex(close.index).ffill().fillna(0.0)
    strat_ret = position.shift(1).fillna(0.0) * _log_ret(close)
    return float(close.iloc[0]) * np.exp(strat_ret.cumsum())


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCIA DEL SEMÁFORO (versiones corregidas de tcn.py)
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(agent, z_sequences, dates, device, coverage=COVERAGE):
    """
    Decisión por cuantiles de la señal alcista s = P(BUY) - P(SELL):
      s alto -> BUY, s bajo -> SELL, intermedio -> HOLD.
    'coverage' fija la fracción de días con señal direccional (top y bottom
    coverage/2). Devuelve también r_pred (cabeza de regresión).
    """
    agent.eval()
    P, R = [], []
    with torch.no_grad():
        for i in range(len(z_sequences)):
            x = torch.as_tensor(z_sequences[i], dtype=torch.float32, device=device).unsqueeze(0)
            logits, rpred = agent(x)
            P.append(F.softmax(logits, dim=-1).squeeze(0).cpu().numpy())
            R.append(float(rpred.reshape(-1)[0].cpu()))
    P = np.asarray(P)                      # columnas: [HOLD, BUY, SELL]
    R = np.asarray(R, dtype=float)
    score = P[:, 1] - P[:, 2]              # señal alcista
    hi = np.quantile(score, 1.0 - coverage / 2.0)
    lo = np.quantile(score, coverage / 2.0)
    actions = np.where(score >= hi, 1, np.where(score <= lo, 2, 0))
    return pd.DataFrame({
        "date":   dates[:len(actions)],
        "action": actions,
        "label":  [ACTION_NAMES[a] for a in actions],
        "p_hold": P[:, 0], "p_buy": P[:, 1], "p_sell": P[:, 2],
        "r_pred": R,
    }).set_index("date")


def apply_trend_gate(signals, df, fast=GATE_FAST, dd_look=GATE_DD_LOOK,
                     dd_thr=GATE_DD_THR):
    """
    Régimen alcista si precio > SMA(fast) Y no hay caída brusca reciente.
      - En régimen alcista:  SELL -> HOLD  (no shortear contra la tendencia).
      - En régimen bajista:  BUY  -> HOLD  (no comprar en caída).
    El override de drawdown deja pasar los SELL en COVID/2022, etc.
    """
    close = df["Close"]
    sma   = close.rolling(fast).mean()
    ret_r = np.log(close / close.shift(dd_look))
    bullish = ((close > sma) & (ret_r > dd_thr)).reindex(signals.index).fillna(False).values
    act = signals["action"].values.copy()
    act[bullish  & (act == 2)] = 0
    act[~bullish & (act == 1)] = 0
    out = signals.copy()
    out["action"] = act
    out["label"]  = [ACTION_NAMES[a] for a in act]
    return out


def semaforo_positions(signals, index, allow_short=ALLOW_SHORT):
    """
    Traduce las señales (esparcidas en fechas de decisión) a una posición diaria.
    BUY -> +1, SELL -> -1 (o 0 si no se permite cortos). HOLD mantiene la
    posición previa (la posición solo cambia con BUY/SELL).
    """
    pos = pd.Series(np.nan, index=index, dtype=float)
    for date, act in signals["action"].items():
        if date not in pos.index:
            continue
        if act == 1:
            pos.loc[date] = 1.0
        elif act == 2:
            pos.loc[date] = -1.0 if allow_short else 0.0
        # HOLD (0): se deja NaN -> ffill conserva la posición anterior
    return pos.ffill().fillna(0.0)


def ma_positions(df_ma_test, allow_short=ALLOW_SHORT):
    """Posición de la estrategia de medias móviles a partir de 'signal'."""
    pos = df_ma_test["signal"].astype(float).copy()
    if not allow_short:
        pos = pos.clip(lower=0.0)          # largo / efectivo
    return pos


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — DATOS
# ══════════════════════════════════════════════════════════════════════════════
def stage_data(ticker, name):
    print("\n" + "=" * 70)
    print(f"[1/4]  DESCARGA E INGENIERÍA DE DATOS — {name} ({ticker})")
    print("=" * 70)
    proc = DataProcessor(ticker=ticker, start_date=START_DATE, end_date=END_DATE)

    if DOWNLOADER == "yfinance":
        # Descarga SIN hilos. El descargador multihilo de yfinance + curl_cffi
        # provoca un ACCESS_VIOLATION (segfault) en Windows; threads=False ayuda,
        # pero si curl_cffi sigue cascando usa DOWNLOADER="requests".
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, threads=False, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"yfinance no devolvió datos para {ticker}.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    else:
        # Descarga directa con requests (sin curl_cffi): evita el segfault.
        df = download_yahoo_prices(ticker, START_DATE, END_DATE)

    df.dropna(inplace=True)
    proc.data = df

    proc.add_features()
    df = proc.data.copy()
    train_df, test_df = proc.split_data(stability_end=STABILITY_END)
    print(f"   Sesiones totales : {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"   Entrenamiento    : {len(train_df)}  (estable, hasta {STABILITY_END})")
    print(f"   Test             : {len(test_df)}  (crisis/posterior)")
    return df, train_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — MEDIAS MÓVILES
# ══════════════════════════════════════════════════════════════════════════════
def stage_moving_averages(df, test_index, name):
    print("\n" + "=" * 70)
    print(f"[2/4]  ESTRATEGIA DE MEDIAS MÓVILES (SMA {MA_SHORT}/{MA_LONG}) — {name}")
    print("=" * 70)
    df_ma = generate_ma_signals(df, short_window=MA_SHORT, long_window=MA_LONG,
                                column="Close", ma_type="SMA")
    df_ma_test = df_ma.loc[test_index]
    counts = df_ma_test["signal"].value_counts().sort_index()
    print("   Distribución de señales en test (1=largo, 0=neutral, -1=corto):")
    for k, v in counts.items():
        print(f"      {int(k):>2} : {v}")
    return df_ma_test


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3a — ENTRENAMIENTO DEL CNN-AE  (genera el .pth de ESTE activo)
# ══════════════════════════════════════════════════════════════════════════════
def stage_train_cnn_ae(train_df, weights_path, name, tag):
    print("\n" + "=" * 70)
    print(f"[3a/4] ENTRENAMIENTO CNN-AE — {name}  ->  {weights_path}")
    print("=" * 70)
    if os.path.exists(weights_path) and not FORCE_RETRAIN_AE:
        print(f"   '{weights_path}' ya existe -> se reutiliza "
              f"(pon FORCE_RETRAIN_AE=True para reentrenar).")
        return

    train_feat = train_df[FEATURE_COLS]
    scaler = MinMaxScaler().fit(train_feat)            # se ajusta SOLO con train
    train_sc = scaler.transform(train_feat)
    X_train = make_windows(train_sc, WINDOW_SIZE)
    print(f"   X_train (ventanas) : {X_train.shape}")

    model = ConvAutoencoder(n_features=len(FEATURE_COLS), seq_len=WINDOW_SIZE)
    history = train_autoencoder(model, X_train, epochs=AE_EPOCHS, device=DEVICE)
    save_model(model, weights_path)
    if SHOW_PLOTS:
        plot_training_history(history)


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3b — SEMÁFORO CNN-AE + TCN
# ══════════════════════════════════════════════════════════════════════════════
def stage_semaforo(df, train_df, test_df, weights_path, name, tag):
    print("\n" + "=" * 70)
    print(f"[3b/4] SEMÁFORO CNN-AE (frozen) + TCN — {name}")
    print("=" * 70)

    # ── Features de contexto para la TCN (además del latente del CNN-AE) ──────
    feat_full = pd.DataFrame(index=df.index)
    feat_full["ret_5"]     = np.log(df["Close"] / df["Close"].shift(5))
    feat_full["ret_20"]    = np.log(df["Close"] / df["Close"].shift(20))
    feat_full["px_sma50"]  = df["Close"] / df["SMA_50"]  - 1.0
    feat_full["px_sma200"] = df["Close"] / df["SMA_200"] - 1.0
    feat_full["rsi"]       = df["RSI_14"] / 100.0
    feat_full["vol"]       = df["Volatilidad"]
    feat_full = feat_full.fillna(0.0)

    # ── Ventanas escaladas de las 5 features base (entrada al CNN-AE) ─────────
    full_df = df[FEATURE_COLS]
    train_in, test_in = full_df.loc[train_df.index], full_df.loc[test_df.index]
    scaler = MinMaxScaler().fit(train_in)
    train_sc, test_sc = scaler.transform(train_in), scaler.transform(test_in)
    X_train, X_test = make_windows(train_sc, WINDOW_SIZE), make_windows(test_sc, WINDOW_SIZE)

    # ── Features de contexto alineadas a cada ventana ────────────────────────
    feat_train, feat_test = feat_full.loc[train_in.index], feat_full.loc[test_in.index]
    fmean, fstd = feat_train.mean(), feat_train.std() + 1e-8
    feat_train_sc = ((feat_train - fmean) / fstd).values.astype(np.float32)
    feat_test_sc  = ((feat_test  - fmean) / fstd).values.astype(np.float32)
    fw_train = feat_train_sc[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(X_train)]
    fw_test  = feat_test_sc[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(X_test)]
    N_FEAT = fw_train.shape[1]

    # ── CNN-AE congelado (el .pth de ESTE activo) ────────────────────────────
    cnn_ae = load_cnn_ae(weights_path, n_features=len(FEATURE_COLS), device=DEVICE)
    Z_DIM = 16 * (WINDOW_SIZE // 2) + N_FEAT          # 240 + N_FEAT
    print(f"   z_dim = {Z_DIM}  (240 latente + {N_FEAT} features de contexto)")

    # ── Secuencias latentes (z) y etiquetas ──────────────────────────────────
    z_train = build_zf_sequences(cnn_ae, X_train, fw_train, TCN_SEQ_LEN, DEVICE)
    z_test  = build_zf_sequences(cnn_ae, X_test,  fw_test,  TCN_SEQ_LEN, DEVICE)

    DEC_OFFSET  = WINDOW_SIZE + TCN_SEQ_LEN - 2
    close_train = df["Close"].loc[train_df.index]
    close_test  = df["Close"].loc[test_df.index]
    fwd_train = forward_returns(close_train, DEC_OFFSET, len(z_train), HORIZON)
    fwd_test  = forward_returns(close_test,  DEC_OFFSET, len(z_test),  HORIZON)
    z_train, z_test = z_train[:len(fwd_train)], z_test[:len(fwd_test)]
    r1_train  = forward_returns(close_train, DEC_OFFSET, len(z_train), 1)   # objetivo regresión
    test_dec_dates = test_df.index[DEC_OFFSET: DEC_OFFSET + len(z_test)]
    print(f"   z_train: {z_train.shape} | z_test: {z_test.shape}")

    # ── Agente multitarea ────────────────────────────────────────────────────
    agent = SemaforoAgent(cnn_ae, z_dim=Z_DIM, tcn_channels=TCN_CHANNELS,
                          tcn_kernel=TCN_KERNEL, tcn_seq_len=TCN_SEQ_LEN,
                          dropout=TCN_DROPOUT, context_dim=64).to(DEVICE)
    n_train = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"   Parámetros entrenables: {n_train:,}")

    print("   Entrenando agente...")
    history, best_ep = train_supervised(
        agent, z_train, fwd_train, r1_train, epochs=AGENT_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, thr=0.0, label_smoothing=LABEL_SMOOTH,
        lambda_reg=LAMBDA_REG, device=DEVICE)

    if SHOW_PLOTS:
        plt.figure(figsize=(10, 3))
        plt.plot(history, alpha=0.7, label="val_acc")
        plt.scatter([best_ep], [history[best_ep]], color="red", zorder=5,
                    label=f"checkpoint ({history[best_ep]:.3f})")
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="azar")
        plt.title(f"Accuracy de validación (agente TCN) — {name}")
        plt.xlabel("Época"); plt.ylabel("val_acc"); plt.legend()
        plt.tight_layout(); plt.savefig(f"tcn_training_curve_{tag}.png", dpi=150); plt.show()

    # ── Inferencia + filtro de tendencia ─────────────────────────────────────
    signals = apply_trend_gate(
        run_inference(agent, z_test, test_dec_dates, DEVICE, coverage=COVERAGE), df)
    print("\n   Distribución de señales del semáforo:")
    print(signals["label"].value_counts().to_string())
    print_metrics(evaluate_signals(signals, fwd_test), HORIZON)

    # ── Regresión del precio al día siguiente vs baseline de persistencia ────
    c = close_test.values
    d_idx = DEC_OFFSET + np.arange(len(signals))
    valid = d_idx + 1 < len(c)
    d_idx = d_idx[valid]
    pred_price  = c[d_idx] * np.exp(signals["r_pred"].values[valid])
    true_price  = c[d_idx + 1]
    naive_price = c[d_idx]                          # "mañana = hoy"
    reg   = regression_metrics(pred_price,  true_price)
    naive = regression_metrics(naive_price, true_price)
    print("\n   — Regresión del precio (día siguiente) —")
    print(f"      Modelo      : R²={reg['R2']:.4f} | RMSE={reg['RMSE']:.2f} | "
          f"MAE={reg['MAE']:.2f} | MAPE={reg['MAPE']:.3f}%")
    print(f"      Persistencia: R²={naive['R2']:.4f} | RMSE={naive['RMSE']:.2f} | "
          f"MAE={naive['MAE']:.2f} | MAPE={naive['MAPE']:.3f}%")

    return signals, close_test


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — MÉTRICAS Y COMPARATIVA ECONÓMICA (de ESTE activo)
# ══════════════════════════════════════════════════════════════════════════════
def stage_metrics(test_close, df_ma_test, signals, name, tag):
    print("\n" + "=" * 70)
    print(f"[4/4]  MÉTRICAS Y COMPARATIVA DE ESTRATEGIAS — {name}")
    print("=" * 70)

    idx = test_close.index

    # Curvas de capital (todas arrancan en el mismo punto -> comparables)
    eq_bh  = test_close.copy()                                    # Buy & Hold del activo
    eq_ma  = equity_from_positions(test_close, ma_positions(df_ma_test))
    eq_sem = equity_from_positions(test_close, semaforo_positions(signals, idx))

    strategies = {
        f"Semáforo (CNN-AE+TCN) — {name}":  eq_sem.values,
        f"Medias Móviles {MA_SHORT}/{MA_LONG} — {name}": eq_ma.values,
        f"Buy & Hold — {name}":             eq_bh.values,
    }
    compare_strategies(strategies, risk_free_rate=RISK_FREE, verbose=True)

    # Criterios de éxito del TFM (benchmark = Buy & Hold del propio activo)
    m_sem = calculate_strategy_metrics(eq_sem.values, f"Semáforo {name}",  RISK_FREE)
    m_ma  = calculate_strategy_metrics(eq_ma.values,  f"MA {name}",        RISK_FREE)
    m_bh  = calculate_strategy_metrics(eq_bh.values,  f"Buy & Hold {name}", RISK_FREE)
    evaluate_tfm_criteria(
        sharpe_semaforo=m_sem["sharpe_ratio"],
        sharpe_sp500=m_bh["sharpe_ratio"],          # benchmark Sharpe = B&H del activo
        mdd_semaforo=m_sem["max_drawdown"],
        mdd_buy_hold=m_bh["max_drawdown"],
        retorno_semaforo=m_sem["cumulative_return"],
        retorno_ma_referencia=m_ma["cumulative_return"],
        verbose=True,
    )

    # Gráfico comparativo de curvas de capital
    plt.figure(figsize=(13, 5))
    plt.plot(idx, eq_sem.values, label="Semáforo (CNN-AE+TCN)", color="#16a34a", linewidth=1.4)
    plt.plot(idx, eq_ma.values,  label=f"Medias Móviles {MA_SHORT}/{MA_LONG}", color="#2563eb", linewidth=1.2)
    plt.plot(idx, eq_bh.values,  label="Buy & Hold", color="#6b7280", linewidth=1.2, alpha=0.85)
    plt.title(f"Curvas de capital — periodo test — {name}")
    plt.xlabel("Fecha"); plt.ylabel("Capital (base = primer cierre)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"equity_curves_{tag}.png", dpi=150, bbox_inches="tight")
    print(f"\n   Gráfico guardado en equity_curves_{tag}.png")
    if SHOW_PLOTS:
        plt.show()

    return {"name": name, "m_sem": m_sem, "m_ma": m_ma, "m_bh": m_bh}


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE UN ACTIVO (las 4 etapas en orden)
# ══════════════════════════════════════════════════════════════════════════════
def run_asset(asset):
    ticker, name = asset["ticker"], asset["name"]
    tag = safe_token(ticker)
    weights_path = f"cnn_ae_weights_{tag}.pth"

    print("\n" + "#" * 70)
    print(f"#  ACTIVO: {name}  ({ticker})")
    print("#" * 70)

    df, train_df, test_df = stage_data(ticker, name)                       # [1]
    df_ma_test            = stage_moving_averages(df, test_df.index, name) # [2]
    stage_train_cnn_ae(train_df, weights_path, name, tag)                  # [3a] -> .pth
    signals, test_close   = stage_semaforo(df, train_df, test_df,
                                           weights_path, name, tag)        # [3b]
    result = stage_metrics(test_close, df_ma_test, signals, name, tag)     # [4]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN COMPARATIVO ENTRE ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════
def print_cross_asset_summary(results):
    print("\n" + "#" * 70)
    print("#  RESUMEN COMPARATIVO ENTRE ACTIVOS  (estrategia Semáforo vs Buy & Hold)")
    print("#" * 70)

    rows = []
    for r in results:
        s, b = r["m_sem"], r["m_bh"]
        rows.append({
            "Activo":            r["name"],
            "Ret. Semáforo":     f"{s['cumulative_return']:.2%}",
            "Ret. B&H":          f"{b['cumulative_return']:.2%}",
            "Sharpe Semáforo":   f"{s['sharpe_ratio']:.3f}",
            "Sharpe B&H":        f"{b['sharpe_ratio']:.3f}",
            "MDD Semáforo":      f"{s['max_drawdown']:.2%}",
            "MDD B&H":           f"{b['max_drawdown']:.2%}",
        })
    df_summary = pd.DataFrame(rows)
    print("\n" + df_summary.to_string(index=False))

    try:
        df_summary.to_csv("resumen_activos.csv", index=False)
        print("\nResumen guardado en resumen_activos.csv")
    except Exception as e:
        print(f"\n(no se pudo guardar el CSV: {e})")


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("#" * 70)
    print("#  PIPELINE TFM — Semáforo bursátil (CNN-AE + TCN)")
    print(f"#  Activos: {', '.join(a['name'] for a in ASSETS)}")
    print(f"#  Dispositivo: {DEVICE}")
    print("#" * 70)

    results = []
    for asset in ASSETS:               # secuencial: primero uno, luego otro, etc.
        results.append(run_asset(asset))
        # Liberar memoria entre activos (evita agotar RAM/VRAM con 3 entrenamientos)
        plt.close("all")
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print_cross_asset_summary(results)

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETO (todos los activos)")
    print("#" * 70)
    return results


if __name__ == "__main__":
    main()