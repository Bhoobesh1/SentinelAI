"""
SentinelAI — XGBoost multiclass attack classifier.

Classifies each event into one of cfg.CLASSIFIER_CLASSES:
    normal, brute_force, impossible_travel, credential_stuffing,
    lateral_movement, device_spoofing, low_slow_exfiltration, insider_drift

Uses the SAME per-event tabular features as the autoencoder
(cfg.MODEL_FEATURE_COLUMNS) -- this is a per-event classifier, not a
sequence model, and is trained completely independently of the
autoencoder. The two signals are combined later in the hybrid risk
engine (Stage 6).

Class imbalance is handled via balanced sample weighting (inverse class
frequency), NOT oversampling -- oversampling sequential security events
would duplicate temporal patterns and distort the sequence-level
autoencoder's story if reused elsewhere, and generally risks the model
memorizing near-duplicate synthetic minority examples.
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel_ai.classifier")


def prepare_labels(df: pd.DataFrame) -> tuple:
    """Fit a LabelEncoder over the FIXED class list (not just classes seen
    in training data), so the encoding is stable regardless of which
    attack types happen to appear in a given split."""
    encoder = LabelEncoder()
    encoder.fit(cfg.CLASSIFIER_CLASSES)
    y = encoder.transform(df["attack_type"].values)
    return y, encoder


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, num_classes: int):
    """Train via the low-level xgb.train/Booster API rather than the
    sklearn XGBClassifier wrapper. This matters here specifically because
    a rare attack type (e.g. insider_drift, injected only in a narrow
    mid-simulation window) can end up ENTIRELY ABSENT from the
    chronological training split by chance -- the sklearn wrapper refuses
    to fit if not every class 0..num_classes-1 appears in y_train, but
    the Booster API only needs `num_class` to build a correctly-shaped
    softmax output, regardless of which classes were actually observed."""
    import xgboost as xgb

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": cfg.XGB_MAX_DEPTH,
        "eta": cfg.XGB_LEARNING_RATE,
        "subsample": cfg.XGB_SUBSAMPLE,
        "colsample_bytree": cfg.XGB_COLSAMPLE_BYTREE,
        "eval_metric": "mlogloss",
        "seed": cfg.RANDOM_SEED,
    }
    booster = xgb.train(params, dtrain, num_boost_round=cfg.XGB_N_ESTIMATORS)
    return booster


def evaluate_classifier(model, X, y_true, encoder: LabelEncoder) -> dict:
    import xgboost as xgb

    dmatrix = xgb.DMatrix(X)
    y_proba = model.predict(dmatrix)  # (n, num_class) since objective=multi:softprob
    y_pred = np.argmax(y_proba, axis=1)

    present_labels = sorted(set(y_true) | set(y_pred))
    target_names = encoder.inverse_transform(present_labels)

    report_str = classification_report(
        y_true, y_pred, labels=present_labels, target_names=target_names,
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    per_class = {
        target_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(present_labels))
    }

    return {
        "report_str": report_str,
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(target_names),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_artifacts(model, encoder: LabelEncoder, feature_cols: list, metrics: dict):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    model.save_model(cfg.XGB_MODEL_PATH)  # Booster.save_model
    with open(cfg.XGB_LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    meta = {
        "feature_cols": feature_cols,
        "classes": list(encoder.classes_),
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "test_per_class": metrics["per_class"],
    }
    with open(cfg.XGB_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_artifacts():
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(cfg.XGB_MODEL_PATH)
    with open(cfg.XGB_LABEL_ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(cfg.XGB_META_PATH, "r") as f:
        meta = json.load(f)
    return model, encoder, meta
