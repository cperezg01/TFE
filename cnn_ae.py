import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import DataLoader, TensorDataset

# Features que usamos del DataProcessor — seleccionamos 5 que capturan
# retorno, riesgo, momentum y tendencia
FEATURE_COLS = ["Log_Ret", "Volatilidad", "RSI_14", "MACD_12_26_9", "SMA_50"]


class ConvAutoencoder(nn.Module):
    """
    Autoencoder Convolucional (CNN-AE) para detección de anomalías en series
    temporales financieras.

    Principio: se entrena SOLO con datos de periodos estables (pre-2020).
    La red aprende a reconstruir patrones "normales". Ante una crisis, el error
    de reconstrucción (MSE) sube — esa subida es la señal de anomalía.

    Args:
        n_features: número de características (columnas del tensor de entrada)
        seq_len:    longitud de la ventana temporal (días por muestra, default 30)
    """

    def __init__(self, n_features: int, seq_len: int = 30):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        # ── ENCODER ───────────────────────────────────────────────────────────
        # Conv1d(in_channels, out_channels, kernel_size, ...)
        # Cada filtro "escanea" ventanas de 3 días consecutivos buscando patrones.
        # BatchNorm1d estabiliza el entrenamiento normalizando las activaciones.
        # stride=2 en la segunda capa comprime la longitud temporal: 30 → 15.
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # ── DECODER ───────────────────────────────────────────────────────────
        # ConvTranspose1d es la "convolución inversa": amplía la dimensión temporal.
        # output_padding=1 asegura que 15 → 30 exactamente (paridad con el encoder).
        # Sigmoid al final porque los datos están en [0, 1] tras MinMaxScaler.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, n_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Proyecta x al espacio latente.

        PyTorch Conv1d espera (batch, canales, longitud), pero los tensores del
        DataProcessor llegan como (batch, seq_len, n_features) → permutamos.
        """
        return self.encoder(x.permute(0, 2, 1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruye desde el espacio latente al espacio original."""
        return self.decoder(z).permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ── ENTRENAMIENTO ──────────────────────────────────────────────────────────────

def train_autoencoder(
    model: ConvAutoencoder,
    X_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str | None = None,
) -> dict:
    """
    Entrena el autoencoder con datos de estabilidad.

    Usa un pequeño conjunto de validación interno para monitorizar que el modelo
    generaliza (no sobreajusta el conjunto de entrenamiento).

    Args:
        model:      instancia de ConvAutoencoder (sin inicializar en GPU aún)
        X_train:    array numpy (muestras, seq_len, n_features) — solo datos estables
        epochs:     número de pasadas completas por los datos
        batch_size: muestras por mini-batch
        lr:         tasa de aprendizaje del optimizador Adam
        val_split:  fracción de X_train reservada para validación
        device:     'cuda', 'mps', 'cpu' — si None, se detecta automáticamente

    Returns:
        dict con 'train_loss' y 'val_loss' (listas de floats por época)
    """
    if device is None:
        device = _auto_device()
    print(f"Entrenando en: {device}")
    model.to(device)

    # División entrenamiento / validación (temporal, sin shuffle)
    n_val = int(len(X_train) * val_split)
    X_val_np = X_train[-n_val:]
    X_tr_np  = X_train[:-n_val]

    X_tr  = torch.tensor(X_tr_np,  dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)

    loader_tr  = DataLoader(TensorDataset(X_tr),  batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # ── fase de entrenamiento ─────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for (batch,) in loader_tr:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── fase de validación ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in loader_val:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch).item()

        avg_tr  = train_loss / len(loader_tr)
        avg_val = val_loss  / len(loader_val)
        history["train_loss"].append(avg_tr)
        history["val_loss"].append(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}  |  train={avg_tr:.6f}  val={avg_val:.6f}")

    return history


# ── INFERENCIA Y DETECCIÓN ─────────────────────────────────────────────────────

def compute_reconstruction_errors(
    model: ConvAutoencoder,
    X: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Calcula el MSE de reconstrucción para cada ventana de X.

    MSE alto → el mercado no se parece a lo aprendido → posible anomalía.

    Returns:
        array 1-D de longitud len(X) con el error por muestra
    """
    if device is None:
        device = _auto_device()
    model.eval()
    model.to(device)

    errors = []
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            # MSE por muestra: promedio sobre (seq_len, n_features)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors)


def calibrate_thresholds(train_errors: np.ndarray, p_amber: float = 90, p_red: float = 97) -> dict:
    """
    Calcula los umbrales del semáforo usando percentiles del error en datos de entrenamiento.

    Lógica: si el modelo ve datos parecidos a los de entrenamiento (mercado normal),
    el MSE debería estar dentro del rango habitual. Definimos "fuera del rango" como
    por encima del percentil 90 (ámbar) o 97 (rojo).

    Args:
        train_errors: errores de reconstrucción calculados sobre el set de entrenamiento
        p_amber:      percentil para el umbral verde/ámbar (default 90)
        p_red:        percentil para el umbral ámbar/rojo  (default 97)

    Returns:
        dict con 'green_max' y 'amber_max'
    """
    thresholds = {
        "green_max": float(np.percentile(train_errors, p_amber)),
        "amber_max": float(np.percentile(train_errors, p_red)),
    }
    print(f"Umbrales calibrados — Verde<{thresholds['green_max']:.6f}  Ámbar<{thresholds['amber_max']:.6f}")
    return thresholds


def classify_signals(errors: np.ndarray, thresholds: dict) -> list[str]:
    """Asigna verde / ámbar / rojo a cada muestra según su error de reconstrucción."""
    result = []
    for e in errors:
        if e < thresholds["green_max"]:
            result.append("verde")
        elif e < thresholds["amber_max"]:
            result.append("ámbar")
        else:
            result.append("rojo")
    return result


# ── VISUALIZACIÓN ──────────────────────────────────────────────────────────────

def plot_training_history(history: dict) -> None:
    """Muestra la curva de pérdida durante el entrenamiento."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss", linestyle="--")
    ax.set_title("Evolución de la pérdida (MSE) durante el entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_semaforo(dates, errors: np.ndarray, thresholds: dict, title: str = "Semáforo de Riesgo") -> None:
    """
    Visualiza el error de reconstrucción a lo largo del tiempo con las zonas del semáforo.

    Args:
        dates:      índice de fechas alineado con errors (len = len(errors))
        errors:     array 1-D de MSE por muestra
        thresholds: dict devuelto por calibrate_thresholds()
    """
    g_max = thresholds["green_max"]
    a_max = thresholds["amber_max"]
    y_max = float(errors.max()) * 1.15

    fig, ax = plt.subplots(figsize=(15, 5))

    # Zonas de color
    ax.axhspan(0,     g_max, alpha=0.12, color="green",  label="Zona verde (normal)")
    ax.axhspan(g_max, a_max, alpha=0.12, color="orange", label="Zona ámbar (precaución)")
    ax.axhspan(a_max, y_max, alpha=0.12, color="red",    label="Zona roja (anomalía)")

    # Líneas de umbral
    ax.axhline(g_max, color="green",  linestyle="--", linewidth=1)
    ax.axhline(a_max, color="red",    linestyle="--", linewidth=1)

    # Error de reconstrucción
    ax.plot(dates, errors, color="steelblue", linewidth=1, label="Error de reconstrucción (MSE)")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ── PERSISTENCIA ───────────────────────────────────────────────────────────────

def save_model(model: ConvAutoencoder, path: str = "cnn_ae_weights.pth") -> None:
    """Guarda los pesos del modelo entrenado."""
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en {path}")


def load_model(n_features: int, path: str = "cnn_ae_weights.pth", seq_len: int = 30) -> ConvAutoencoder:
    """Carga un modelo previamente guardado."""
    model = ConvAutoencoder(n_features=n_features, seq_len=seq_len)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"Modelo cargado desde {path}")
    return model


# ── UTILIDADES ─────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    """Detecta el mejor dispositivo disponible (GPU CUDA, Apple MPS, o CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
