"""
SentinelAI — Real-time streaming scorer.

Proves the "near real-time" requirement empirically rather than just on
paper: every stateful component in this system (EntityRunningState,
EntityProfiler) was ALREADY built as an incremental, single-pass,
causal algorithm -- the batch pipeline just happens to drive it with a
for-loop over a static CSV. This module drives the EXACT SAME
`compute_event_features()` function (see src/features/feature_engineering.py)
one live event at a time, proving that no architectural change is needed
to go from batch to streaming -- only the event SOURCE changes (a Kafka
consumer instead of a pandas DataFrame iterator).

Per-event pipeline (identical logic to the batch path, just invoked
one row at a time):
    1. compute_event_features()      <- same causal feature code as batch
    2. append to a per-entity rolling window (deque, maxlen=SEQUENCE_LENGTH)
    3. once the window is full: scale + autoencoder -> sequence error
    4. XGBoost predict_proba on this one event -> attack confidence
    5. risk_engine.compute_risk_components() -> 0-100 risk score
    6. risk_engine.compute_hybrid_anomaly_score() -> 0-1 anomaly score

Latency is measured per event so throughput/scalability claims are
backed by actual numbers, not estimates.
"""

import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg
from src.features.feature_engineering import compute_event_features
from src.profiling.entity_profiler import EntityProfiler
from src.models.risk_engine import (
    compute_risk_components, compute_hybrid_anomaly_score, risk_level,
)


class StreamingScorer:
    """Holds all state needed to score a live event stream one event at a
    time: per-entity causal feature state, the hierarchical profiler, a
    rolling per-entity feature window for sequence scoring, and the
    already-trained models (loaded once, reused for every event)."""

    def __init__(self, ae_model, ae_scaler, ae_meta, xgb_booster, xgb_encoder, xgb_meta):
        self.entity_states: dict = {}
        self.profiler = EntityProfiler()

        self.ae_model, self.ae_scaler, self.ae_meta = ae_model, ae_scaler, ae_meta
        self.xgb_booster, self.xgb_encoder, self.xgb_meta = xgb_booster, xgb_encoder, xgb_meta

        self.feature_cols = xgb_meta["feature_cols"]
        self.seq_len = ae_meta["sequence_length"]
        self.log1p_cols = ae_meta["log1p_columns"]

        # Rolling per-entity window of (log1p-applied, unscaled) feature
        # vectors, for sequence-based scoring once enough context exists.
        self.feature_windows: dict = {}

        self.events_processed = 0
        self.latencies_ms = []

    def _feature_vector(self, features: dict) -> np.ndarray:
        return np.array([features[c] for c in self.feature_cols], dtype=np.float32)

    def _log1p_vector(self, raw_vector: np.ndarray) -> np.ndarray:
        v = raw_vector.copy()
        for i, col in enumerate(self.feature_cols):
            if col in self.log1p_cols:
                v[i] = np.log1p(max(v[i], 0.0))
        return v

    def score_event(self, event: dict) -> dict:
        """event: dict with the same fields as one row of raw events.csv
        (entity_id, entity_type, timestamp, geo_location, device_fingerprint,
        auth_method, auth_success, resource_accessed, session_duration,
        command_sequence, event_id)."""
        t0 = time.perf_counter()

        entity_id = event["entity_id"]
        cmds = [c for c in str(event.get("command_sequence", "") or "").split("|") if c]

        # ---- Step 1: causal feature computation (SAME function as batch) ----
        features = compute_event_features(
            entity_states=self.entity_states, profiler=self.profiler,
            entity_id=entity_id, entity_type=event["entity_type"], ts=event["timestamp"],
            geo=event["geo_location"], device=event["device_fingerprint"], method=event["auth_method"],
            auth_success=bool(event["auth_success"]), resource=event["resource_accessed"],
            duration=float(event["session_duration"]), cmds=cmds,
        )

        raw_vector = self._feature_vector(features)

        # ---- Step 2/3: rolling sequence window + autoencoder ----
        window = self.feature_windows.setdefault(entity_id, deque(maxlen=self.seq_len))
        window.append(self._log1p_vector(raw_vector))

        sequence_error = None
        if len(window) == self.seq_len:
            X = np.stack(window).reshape(1, self.seq_len, len(self.feature_cols))
            X_scaled = self.ae_scaler.transform(X.reshape(-1, len(self.feature_cols))).reshape(X.shape).astype(np.float32)
            # Direct call (not .predict()) avoids Keras's per-call retracing
            # overhead, which dominates latency for single-sample inference --
            # a production deployment would use a dedicated low-latency
            # serving layer (TF Serving / ONNX Runtime) for this same reason.
            recon = self.ae_model(X_scaled, training=False).numpy()
            sequence_error = float(np.mean(np.square(X_scaled - recon)))

        # ---- Step 4: XGBoost classifier (single event, no scaling needed) ----
        import xgboost as xgb
        dmatrix = xgb.DMatrix(raw_vector.reshape(1, -1))
        proba = self.xgb_booster.predict(dmatrix)[0]
        classes = list(self.xgb_encoder.classes_)
        normal_idx = classes.index("normal")
        predicted_idx = int(np.argmax(proba))
        predicted_class = classes[predicted_idx]
        attack_confidence = 1.0 - proba[normal_idx]

        # ---- Step 5/6: risk engine (plain dict works identically to a pd.Series here) ----
        row = dict(features)
        row["attack_confidence"] = attack_confidence
        row["sequence_reconstruction_error"] = sequence_error
        row["_ae_threshold"] = self.ae_meta["threshold"]

        risk = compute_risk_components(row)
        hybrid_score = compute_hybrid_anomaly_score(row)
        risk["risk_level"] = risk_level(risk["risk_score"])
        risk["hybrid_anomaly_score"] = hybrid_score
        risk["predicted_attack_type"] = predicted_class
        risk["attack_confidence"] = attack_confidence
        risk["event_id"] = event.get("event_id")
        risk["entity_id"] = entity_id
        risk["timestamp"] = str(event["timestamp"])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.latencies_ms.append(elapsed_ms)
        self.events_processed += 1
        risk["_processing_latency_ms"] = round(elapsed_ms, 3)

        return risk

    def latency_summary(self) -> dict:
        if not self.latencies_ms:
            return {}
        arr = np.array(self.latencies_ms)
        return {
            "events_processed": self.events_processed,
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
            "throughput_events_per_sec_single_core": float(1000.0 / np.mean(arr)),
        }
