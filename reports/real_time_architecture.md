# SentinelAI — Real-Time Streaming Architecture & Scalability

## 1. Why the current codebase is NOT a batch-only afterthought

A common failure mode when adapting a batch ML pipeline to streaming is
discovering that the feature engineering was written assuming random
access to a full historical DataFrame (`df.groupby()`, `df.rolling()`,
`shift(-1)` lookaheads) — which cannot run online at all without a full
rewrite.

SentinelAI does not have this problem. From Stage 2 onward, every
stateful computation was deliberately built as an **incremental, causal,
single-pass algorithm**:

- `EntityRunningState` (`src/profiling/entity_profiler.py`) maintains
  running counters, sets, and deques (devices seen, locations seen,
  command vocabulary, time-windowed event counts) that update in O(1)
  per event.
- `EntityProfiler` maintains running sums (`session_sum`,
  `session_sumsq`, circular-mean accumulators) for the hierarchical
  entity → entity-type → global baseline blend — again O(1) per event.
- `compute_event_features()` (`src/features/feature_engineering.py`,
  extracted during this hardening pass) reads current state, computes
  one event's feature vector, and only THEN commits the update —
  the exact causality discipline a real-time system requires (an event
  must never see itself or the future).

**The batch pipeline (`build_features.py`) and the streaming demo
(`scripts/simulate_streaming.py`) call the literal same function.** This
was verified with a byte-identical regression test after extracting the
function — batch and streaming are not two implementations that need to
be kept in sync; they are one implementation driven two different ways.

This means the only thing that changes going from batch to production
streaming is the **event source** (a pandas DataFrame iterator today, a
Kafka/Kinesis consumer in production) and the **state store** (an
in-memory Python dict today, Redis in production for durability and
multi-worker sharing).

## 2. Measured current-code latency (not an estimate)

`scripts/simulate_streaming.py` replays real events one at a time through
the full pipeline (feature computation -> autoencoder -> XGBoost -> risk
engine) and measures actual wall-clock latency per event.

Steady-state result (3,000 events, single Python process, single CPU
core, no GPU, no model-serving layer, no batching):

| Metric | Value |
|---|---|
| Mean latency/event | ~75 ms |
| P50 | ~88 ms |
| P95 | ~107 ms |
| P99 | ~118 ms |
| Max | ~185 ms |
| Throughput (single core) | ~13 events/sec |

Two honest observations from this measurement:

1. **Latency is bimodal.** Events for entities that haven't yet
   accumulated a full 10-event sequence window skip the autoencoder step
   entirely and score in ~1-2 ms. Once an entity's window fills
   (typically within its first few events), every subsequent event
   costs ~90-100 ms -- dominated almost entirely by the GRU autoencoder
   call.
2. **The ~90-100 ms is a Keras eager-mode inference artifact, not an
   inherent cost of the model itself.** A 10-timestep x 25-feature GRU
   forward pass is computationally tiny; the overhead comes from
   `model.__call__` / per-call cost in a plain Python loop, not from the
   model's actual math. This points directly at the production fix: a
   dedicated low-latency serving layer (below) eliminates most of this
   overhead by batching concurrent requests and keeping the graph warm.

Even taking the worst-case 185 ms at face value, this is comfortably
"near real-time" for a SOC use case -- nobody needs sub-millisecond
detection of impossible travel or brute force. The throughput number
(13 events/sec on one unoptimized core) is the one that needs a
scaling story, which Section 4 provides.

## 3. Production streaming architecture

```
+-------------+     +------------------+     +-----------------------+
|  Event       |---->|  Kafka / Kinesis  |---->|  Stream Consumer      |
|  Sources     |     |  (partitioned by  |     |  Group (N workers,    |
|  (auth logs, |     |   entity_id)      |     |  one partition each)  |
|  API gateway,|     +------------------+     +-----------+-----------+
|  IoT/edge)   |                                           |
+-------------+                                           v
                                            +---------------------------+
                                            | Per-worker in-process:     |
                                            |  compute_event_features()  |
                                            |  (reads/writes Redis for   |
                                            |   entity state, NOT local  |
                                            |   dict, so any worker can  |
                                            |   take over a partition)   |
                                            +-----------+---------------+
                                                         v
                                            +---------------------------+
                                            | Model Serving Layer        |
                                            | (TF Serving / ONNX Runtime |
                                            |  / Triton) -- autoencoder  |
                                            |  + XGBoost, micro-batched  |
                                            |  across concurrent events  |
                                            |  from all workers          |
                                            +-----------+---------------+
                                                         v
                                            +---------------------------+
                                            | Risk Engine (stateless,    |
                                            | pure function -- scales    |
                                            | trivially, no changes      |
                                            | needed from current code)  |
                                            +-----------+---------------+
                                                         |
                                  +----------------------+--------------------+
                                  v                                          v
                      +--------------------+                  +----------------------+
                      | Alert Topic (Kafka)|                  | Attack-Chain          |
                      | -> SIEM / webhook / |                  | Correlation (small    |
                      |    SOC dashboard    |                  | stateful service,     |
                      |    (websocket push) |                  | same time-window      |
                      +--------------------+                  | logic as Stage 9)     |
                                                                +----------------------+
```

### Component mapping from current code to production

| Current (batch/demo) | Production equivalent | Code changes needed |
|---|---|---|
| `pandas.read_csv(events.csv)` | Kafka/Kinesis consumer | Swap the event source only |
| `entity_states: dict` (in-process) | Redis hash per entity_id | Serialize `EntityRunningState`/`EntityProfiler` fields to/from Redis; logic unchanged |
| `ae_model(X, training=False)` in-process | TF Serving / ONNX Runtime endpoint, micro-batched | Wrap model calls in an RPC client; no feature-engineering changes |
| `xgb_booster.predict()` in-process | Same serving layer, or kept in-process (XGBoost inference is already sub-millisecond) | Optional |
| `compute_risk_components()` | Unchanged -- already a pure, stateless function | None |
| `correlate_chains.py` (batch, whole-dataset) | A small stateful service holding a rolling time-window per entity (same `CHAIN_TIME_WINDOW_MINUTES` logic) | Convert the offline sweep into an incremental sliding-window service |

## 4. Scalability

### 4.1 Natural sharding key: `entity_id`
Every stateful computation in this system -- `EntityRunningState`,
`EntityProfiler`'s per-entity component, sequence windows, attack-chain
time-windows -- is scoped to a single entity. This means the event
stream can be partitioned by `entity_id` (Kafka's native partitioning
model) with **zero cross-partition coordination required** for feature
computation or sequence scoring. N Kafka partitions -> N independent
consumer workers -> near-linear throughput scaling.

Entity-type and global baseline updates (`EntityProfiler`'s pooled
levels) are the one piece of state that IS shared across all entities.
In production this would be maintained by a small number of dedicated
aggregator workers consuming a fan-out of all partitions at low
frequency (e.g., a periodic batch merge every few seconds), not on the
per-event hot path -- entity-type/global baselines change slowly by
nature, so eventual consistency here is an acceptable trade-off that
doesn't compromise entity-level detection latency.

### 4.2 Horizontal throughput estimate
At ~75 ms/event average latency (current, unoptimized, single-core),
a naive 1:1 mapping already gives ~13 events/sec/worker. With:
- **Model serving micro-batching** (batching concurrent requests from
  multiple workers into one optimized inference call): expect
  a meaningful reduction in the dominant ~90ms autoencoder cost, based
  on typical batching gains for small sequence models.
- **N worker processes** (one per Kafka partition): near-linear scaling.

A conservative estimate: 20 workers x ~50-100 events/sec/worker
(post-batching) ~= **1,000-2,000 events/sec**, which covers most
mid-to-large enterprise SOC ingestion volumes. Scaling further is a
matter of adding partitions/workers, not re-architecting.

### 4.3 State store choice: Redis
`EntityRunningState` and `EntityProfiler`'s per-entity data (device
sets, location sets, running sums, command vocabulary, time-windowed
deques) are small per entity (a few KB) and map naturally onto Redis
hashes/sets. Redis gives: sub-millisecond state read/write, built-in
TTL for entities that go inactive, and -- critically -- lets ANY worker
pick up ANY partition after a rebalance without losing that entity's
behavioral history (unlike an in-process Python dict, which dies with
the worker).

### 4.4 Fault tolerance / exactly-once considerations
- **Checkpointing**: Kafka consumer offsets committed only after a
  successful risk score is produced AND written downstream -- standard
  at-least-once semantics, acceptable here since re-scoring a duplicate
  event is idempotent (state updates would double-count, so dedup by
  `event_id` before committing state changes in production).
- **Out-of-order events**: the current design assumes a
  globally-sorted stream. Real deployments see some clock skew /
  network delay causing near-real-time events to arrive slightly
  out of order. A small watermark buffer (e.g., 5-10 seconds, holding
  events before committing them into entity state) absorbs this
  without materially affecting detection latency.
- **Worker failure**: because state lives in Redis (not in-process),
  a failed worker's partition can be picked up by another worker with
  zero loss of behavioral history -- only in-flight events need
  reprocessing.

## 5. Summary

No architectural rewrite is needed to go from the current batch
demonstration to a production streaming deployment -- the causal,
incremental design was already there from Stage 2 onward, and this was
proven (not just claimed) by extracting the shared feature-computation
function and running it successfully in both a batch script and a live
per-event streaming simulation with measured latency. The remaining
production work is infrastructure (Kafka, Redis, a model-serving layer)
around an already-correct algorithmic core, not new feature-engineering
or modeling work.
