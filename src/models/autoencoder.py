"""
SentinelAI — GRU sequence autoencoder for behavioral anomaly detection.

Pipeline:
    features.csv + ground_truth/labels.csv (joined ONLY for split/train
    selection and evaluation -- never as a model input feature)
        -> chronological train/val/test split (by event timestamp)
        -> per-entity sliding windows of SEQUENCE_LENGTH events
        -> StandardScaler fit on TRAIN sequences only
        -> GRU encoder-decoder trained on NORMAL-only train sequences
        -> reconstruction-error threshold selected on VALIDATION data only
        -> persisted: model, scaler, threshold, feature column order

A sequence's "anchor event" is its last timestep; the reconstruction
error for a sequence is treated as the sequence-anomaly score for that
anchor event. A window's label (for train-set filtering and evaluation)
is the max of is_anomaly across all timesteps in the window: if ANY
event in the trailing context is anomalous, the window is not "purely
normal" behavior and is excluded from autoencoder training.
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel_ai.autoencoder")


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, ts_col: str = "timestamp"):
    """Return (train_df, val_df, test_df) split purely by time cutoffs,
    computed from the overall min/max timestamp -- no shuffling, no
    per-entity leakage across the boundary."""
    ts = pd.to_datetime(df[ts_col])
    t_min, t_max = ts.min(), ts.max()
    total_seconds = (t_max - t_min).total_seconds()
    train_cutoff = t_min + pd.Timedelta(seconds=total_seconds * cfg.TRAIN_FRAC)
    val_cutoff = t_min + pd.Timedelta(seconds=total_seconds * (cfg.TRAIN_FRAC + cfg.VAL_FRAC))

    train_df = df[ts <= train_cutoff].copy()
    val_df = df[(ts > train_cutoff) & (ts <= val_cutoff)].copy()
    test_df = df[ts > val_cutoff].copy()
    return train_df, val_df, test_df, (train_cutoff, val_cutoff)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int = cfg.SEQUENCE_LENGTH):
    """Build sliding-window sequences PER ENTITY, never crossing entity
    boundaries. Windows do not cross split boundaries either, since this
    is called separately per split.

    Returns: X (n_seq, seq_len, n_features), y (n_seq,) window-level label
             (max of is_anomaly across the window), anchor_event_ids (list),
             anchor_entity_ids (list)
    """
    X, y, anchor_event_ids, anchor_entity_ids = [], [], [], []
    skipped_entities = 0

    for entity_id, g in df.groupby("entity_id"):
        g = g.sort_values("timestamp")
        if len(g) < seq_len:
            skipped_entities += 1
            continue
        feats = g[feature_cols].values.astype(np.float32)
        labels = g["is_anomaly"].values
        event_ids = g["event_id"].values

        for i in range(seq_len - 1, len(g)):
            window = feats[i - seq_len + 1: i + 1]
            window_label = int(labels[i - seq_len + 1: i + 1].max())
            X.append(window)
            y.append(window_label)
            anchor_event_ids.append(event_ids[i])
            anchor_entity_ids.append(entity_id)

    X = np.stack(X) if X else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(y, dtype=int)
    logger.info(f"Built {len(X)} sequences ({skipped_entities} entities skipped for having < {seq_len} events).")
    return X, y, anchor_event_ids, anchor_entity_ids


def apply_log1p(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = np.log1p(np.clip(df[c].values, a_min=0, a_max=None))
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_autoencoder(seq_len: int, n_features: int,
                      gru_units: int = cfg.AE_GRU_UNITS,
                      latent_dim: int = cfg.AE_LATENT_DIM,
                      learning_rate: float = cfg.AE_LEARNING_RATE):
    """GRU encoder-decoder. Chosen over LSTM per project spec ("GRU if
    easier/more stable") -- fewer gates, faster/more stable convergence
    at this dataset scale, negligible capacity difference here."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(seq_len, n_features), name="event_sequence")
    encoded = layers.GRU(gru_units, return_sequences=True, name="encoder_gru_1")(inputs)
    encoded = layers.GRU(latent_dim, return_sequences=False, name="encoder_latent")(encoded)

    repeated = layers.RepeatVector(seq_len, name="repeat_latent")(encoded)
    decoded = layers.GRU(gru_units, return_sequences=True, name="decoder_gru_1")(repeated)
    outputs = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(decoded)

    model = models.Model(inputs, outputs, name="sentinel_gru_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def reconstruction_errors(model, X: np.ndarray) -> np.ndarray:
    """Per-sequence mean squared reconstruction error."""
    if len(X) == 0:
        return np.array([])
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=(1, 2))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(model, scaler, threshold: float, feature_cols: list, seq_len: int):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    model.save(cfg.AUTOENCODER_MODEL_PATH)
    with open(cfg.AUTOENCODER_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    meta = {
        "threshold": float(threshold),
        "feature_cols": feature_cols,
        "sequence_length": seq_len,
        "log1p_columns": cfg.LOG1P_COLUMNS,
    }
    with open(cfg.AUTOENCODER_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_artifacts():
    import tensorflow as tf
    model = tf.keras.models.load_model(cfg.AUTOENCODER_MODEL_PATH)
    with open(cfg.AUTOENCODER_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(cfg.AUTOENCODER_META_PATH, "r") as f:
        meta = json.load(f)
    return model, scaler, meta
