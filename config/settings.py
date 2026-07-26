"""
SentinelAI — Central configuration.

All tunable knobs for data generation, feature engineering, modeling,
risk scoring, and evaluation live here so behavior is reproducible
and easy to explain to judges.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground_truth")
KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_EVENTS_PATH = os.path.join(RAW_DATA_DIR, "events.csv")
GROUND_TRUTH_PATH = os.path.join(GROUND_TRUTH_DIR, "labels.csv")
ENTITY_PROFILES_PATH = os.path.join(PROCESSED_DATA_DIR, "entity_profiles.json")

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
NUM_EVENTS = 20_000            # total raw events to generate (configurable)
NUM_USER_ENTITIES = 55         # human user accounts
NUM_SERVICE_ENTITIES = 8       # service / machine accounts (different rhythm)
NUM_EDGE_DEVICE_ENTITIES = 6   # IoT/OT edge devices (industrial gateway, POS terminal, home IoT hub) --
                               # domain-agnostic per the assessment's framing: same detection pipeline,
                               # a third distinct behavioral rhythm (near-24/7, machine-precise timing,
                               # certificate auth, no human-like variability)
SIMULATION_DAYS = 30           # length of the simulated telemetry window
ANOMALY_RATE = 0.02            # target fraction of events labeled anomalous (0.5%-3%)

SIM_START = "2026-06-01 00:00:00"

# Entity types
ENTITY_TYPES = ["human_user", "service_account", "edge_device"]

# Pool of plausible geo-locations (city, approx lat, lon) used for
# both "home" locations and impossible-travel destinations.
GEO_LOCATIONS = {
    "Chennai":    (13.0827, 80.2707),
    "Bengaluru":  (12.9716, 77.5946),
    "Hyderabad":  (17.3850, 78.4867),
    "Mumbai":     (19.0760, 72.8777),
    "Delhi":      (28.7041, 77.1025),
    "Pune":       (18.5204, 73.8567),
    "Singapore":  (1.3521, 103.8198),
    "Frankfurt":  (50.1109, 8.6821),
    "London":     (51.5072, -0.1276),
    "New_York":   (40.7128, -74.0060),
    "Moscow":     (55.7558, 37.6173),
    "Lagos":      (6.5244, 3.3792),
}

# Resources that entities can access, tagged by sensitivity.
RESOURCES = {
    "/email":              "low",
    "/wiki":               "low",
    "/github":             "medium",
    "/internal-api":       "medium",
    "/hr-portal":          "medium",
    "/finance-db":         "high",
    "/customer-data":      "high",
    "/admin-console":      "high",
    "/source-control-prod":"high",
    "/backup-vault":       "high",
    # IoT/OT "device function" endpoints -- edge_device entities access
    # these almost exclusively, human/service entities essentially never.
    "/telemetry":          "low",
    "/sensor-data":        "medium",
    "/firmware-update":    "high",
}

# password/token/certificate/biometric -- matches the assessment's exact
# auth_method schema wording. sso/mfa_push/mfa_otp/api_key retained for the
# richer human/service-account behavior already built and validated.
AUTH_METHODS = ["password", "sso", "mfa_push", "mfa_otp", "api_key", "certificate", "biometric", "token"]

COMMAND_POOL = [
    "ls", "cd", "cat", "grep", "ssh", "scp", "curl", "wget",
    "git_pull", "git_push", "sudo_su", "whoami", "ps_aux",
    "netstat", "query_db", "export_csv", "zip_archive", "rdp_connect",
]

# ---------------------------------------------------------------------------
# Attack scenario mix (relative weights; must sum to 1.0 across the
# ANOMALY_RATE budget). Insider drift consumes a bucket of *events*
# spread across a gradual window, not single-shot injections.
# ---------------------------------------------------------------------------
ATTACK_TYPES = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_slow_exfiltration",
    "insider_drift",
]

ATTACK_WEIGHTS = {
    "brute_force":            0.20,
    "impossible_travel":      0.15,
    "credential_stuffing":    0.15,
    "lateral_movement":       0.15,
    "device_spoofing":        0.15,
    "low_slow_exfiltration":  0.10,
    "insider_drift":          0.10,
}

assert abs(sum(ATTACK_WEIGHTS.values()) - 1.0) < 1e-9, "ATTACK_WEIGHTS must sum to 1.0"

# ---------------------------------------------------------------------------
# Chronological train/validation/test split (used by ALL models, Stage 4+)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20  # implied, kept for documentation clarity

# ---------------------------------------------------------------------------
# Feature columns fed to ML models (autoencoder + XGBoost).
# Excludes identifiers, timestamp, and the categorical baseline_source label
# (cold_start_weight already encodes that information numerically).
# ---------------------------------------------------------------------------
MODEL_FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "weekend",
    "time_since_last_event", "events_last_10m", "events_last_1h",
    "auth_failure_rate", "failed_attempts_last_10m", "unusual_auth_method",
    "distance_from_previous_login", "geo_velocity", "unusual_location",
    "new_device", "device_change_frequency", "device_consistency_score",
    "resource_novelty", "resource_access_frequency", "sensitive_resource_access",
    "session_duration_deviation",
    "command_novelty", "command_frequency",
    "deviation_from_entity_mean", "deviation_from_entity_std",
    "entity_history_count", "cold_start_weight",
]

# Heavy-tailed, non-negative columns that get a log1p transform before
# scaling, so a handful of extreme values (e.g. brute-force geo_velocity
# spikes) don't dominate the StandardScaler fit.
LOG1P_COLUMNS = [
    "time_since_last_event", "events_last_10m", "events_last_1h",
    "failed_attempts_last_10m", "distance_from_previous_login",
    "geo_velocity", "deviation_from_entity_std", "entity_history_count",
]

# ---------------------------------------------------------------------------
# Sequence autoencoder (Stage 4)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 10          # events of context per sequence
AE_LATENT_DIM = 8
AE_GRU_UNITS = 32
AE_EPOCHS = 30
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3
AE_THRESHOLD_PERCENTILE = 99.0  # percentile of NORMAL validation reconstruction error

AUTOENCODER_MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder.keras")
AUTOENCODER_SCALER_PATH = os.path.join(MODELS_DIR, "autoencoder_scaler.pkl")
AUTOENCODER_META_PATH = os.path.join(MODELS_DIR, "autoencoder_meta.json")

# ---------------------------------------------------------------------------
# XGBoost attack classifier (Stage 5)
# ---------------------------------------------------------------------------
CLASSIFIER_CLASSES = ["normal"] + ATTACK_TYPES  # 8 classes total

XGB_MAX_DEPTH = 6
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_classifier.json")
XGB_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "xgb_label_encoder.pkl")
XGB_META_PATH = os.path.join(MODELS_DIR, "xgb_meta.json")

# ---------------------------------------------------------------------------
# Hybrid anomaly score + dynamic risk engine (Stage 6)
# ---------------------------------------------------------------------------
# Risk score component max-points (must sum to 100)
RISK_MAX_SEQUENCE_ANOMALY = 30
RISK_MAX_BEHAVIOR_DEVIATION = 25
RISK_MAX_ATTACK_CONFIDENCE = 20
RISK_MAX_DEVICE_NOVELTY = 15
RISK_MAX_HISTORICAL_CONTEXT = 10
assert (RISK_MAX_SEQUENCE_ANOMALY + RISK_MAX_BEHAVIOR_DEVIATION + RISK_MAX_ATTACK_CONFIDENCE
        + RISK_MAX_DEVICE_NOVELTY + RISK_MAX_HISTORICAL_CONTEXT) == 100, "Risk components must sum to 100"

RISK_LEVELS = [
    (0, 29, "LOW"),
    (30, 59, "MEDIUM"),
    (60, 79, "HIGH"),
    (80, 100, "CRITICAL"),
]

# Hybrid anomaly score (0-1) weights -- purely statistical/ML, no classifier
HYBRID_WEIGHT_SEQUENCE = 0.40
HYBRID_WEIGHT_BEHAVIOR = 0.35
HYBRID_WEIGHT_NOVELTY = 0.25
assert abs(HYBRID_WEIGHT_SEQUENCE + HYBRID_WEIGHT_BEHAVIOR + HYBRID_WEIGHT_NOVELTY - 1.0) < 1e-9

# Normalization constants
SEQ_ERROR_RATIO_CAP = 2.0     # reconstruction_error / threshold, capped at this multiple
SESSION_Z_CAP = 3.0            # |session_duration_deviation| capped at this many std devs
GEO_VELOCITY_NORM_KMH = 1000.0  # commercial-flight-speed-ish reference for normalizing geo_velocity

RISK_SCORES_PATH = os.path.join(PROCESSED_DATA_DIR, "risk_scores.csv")

# ---------------------------------------------------------------------------
# Attack-chain correlation (Stage 9) -- deterministic, no ML model
# ---------------------------------------------------------------------------
CHAIN_CANDIDATE_RISK_LEVELS = {"MEDIUM", "HIGH", "CRITICAL"}
CHAIN_TIME_WINDOW_MINUTES = 60          # gap threshold to keep same-entity events in one chain
CHAIN_SHARED_IP_LINK_WINDOW_MINUTES = 120  # window for cross-entity linking via shared source_ip

ATTACK_CHAINS_PATH = os.path.join(REPORTS_DIR, "attack_chains.json")

# ---------------------------------------------------------------------------
# RAG cybersecurity knowledge base (Stage 10)
# ---------------------------------------------------------------------------
# Embeddings are TF-IDF (scikit-learn) rather than a neural sentence-
# transformer model. This environment's network is restricted to package
# registries only (no huggingface.co), so a neural-embedding download
# can't be verified to work end-to-end here, and a live demo shouldn't
# depend on downloading model weights at showtime. TF-IDF + FAISS still
# satisfies the full documents -> chunking -> embeddings -> FAISS ->
# retrieval pipeline, fully offline and deterministic. Swapping in
# sentence-transformers later only requires changing `embed_texts()` in
# src/rag/retriever.py.
RAG_TOP_K = 3
RAG_INDEX_PATH = os.path.join(MODELS_DIR, "rag_faiss.index")
RAG_VECTORIZER_PATH = os.path.join(MODELS_DIR, "rag_vectorizer.pkl")
RAG_CHUNKS_PATH = os.path.join(MODELS_DIR, "rag_chunks.json")

# ---------------------------------------------------------------------------
# Concept drift detection -- deterministic, no ML model.
# Distinguishes SUSTAINED behavioral drift (e.g. a legitimate role change)
# from a single anomalous event. Never updates a baseline off one event --
# requires repeated evidence over consecutive days (EWMA-smoothed).
# ---------------------------------------------------------------------------
DRIFT_REFERENCE_DAYS = 10       # entity's first N days of activity establish the reference baseline
DRIFT_RECENT_WINDOW_DAYS = 7    # rolling window compared against the reference
DRIFT_EWMA_ALPHA = 0.3          # smoothing factor; higher = more reactive, lower = more conservative
DRIFT_SCORE_THRESHOLD = 0.35    # composite (0-1) EWMA score above this counts as "elevated"
DRIFT_MIN_CONSECUTIVE_DAYS = 5  # elevated score must persist this many consecutive days to confirm drift

DRIFT_COMPONENT_WEIGHTS = {
    "hour": 0.20,
    "device": 0.30,
    "resource": 0.30,
    "session": 0.20,
}
assert abs(sum(DRIFT_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9

DRIFT_STATUS_PATH = os.path.join(PROCESSED_DATA_DIR, "drift_status.csv")
DRIFT_SUMMARY_PATH = os.path.join(REPORTS_DIR, "drift_summary.json")

# ---------------------------------------------------------------------------
# Automated Threat Response Orchestrator ("SOAR-lite") + Business Impact
# ---------------------------------------------------------------------------
# Illustrative cost-model constants for translating technical risk into a
# business-impact estimate. These are NOT asserted as universal industry
# figures -- every organization should calibrate these to their own breach
# cost history, cyber-insurance data, or a benchmark report they trust.
# The point is the TRANSLATION MECHANISM (technical evidence -> $ estimate),
# not these specific numbers.
RESPONSE_COST_PER_RECORD_USD = 165  # illustrative -- calibrate per organization
RESPONSE_RECORDS_AT_RISK_BY_SENSITIVITY = {"low": 50, "medium": 500, "high": 5000}
RESPONSE_CONTAINMENT_GROWTH_RATE_PER_HOUR = 0.04  # illustrative growth in impact while unaddressed (~2.5x over 24h)

RESPONSE_AUDIT_LOG_PATH = os.path.join(REPORTS_DIR, "response_audit_log.json")
