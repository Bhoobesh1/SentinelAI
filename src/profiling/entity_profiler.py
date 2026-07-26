"""
SentinelAI — Entity behavioral profiling with hierarchical cold-start fallback.

Three baseline levels are maintained simultaneously, all updated causally
(only from events strictly before the one being scored):

    entity level       (e.g. USER_042's own history)
    entity-type level  (all human_user events, or all service_account events)
    global level        (every event, regardless of entity)

When an entity is new, its own statistics are unreliable (or nonexistent),
so we blend toward the entity-type baseline, and further toward the global
baseline if even the entity-type doesn't have enough history yet (e.g. very
early in the simulation window). As the entity accumulates events, the
blend weight shifts toward its own baseline.

This module owns two things:
  * _EntityRunningState — full per-entity running state (devices, locations,
    resources, commands, auth methods, time windows) used directly by
    feature_engineering.py for entity-only signals (new_device,
    unusual_location, auth_failure_rate, etc.) where hierarchical fallback
    doesn't make sense (there is no meaningful "global typical device").
  * EntityProfiler — the hierarchical blending logic for the handful of
    continuous baselines where cold-start fallback DOES make sense:
    session duration, typical login hour, and resource-access frequency.
"""

import math
from collections import Counter, deque, defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENTITY_CONFIDENCE_K = 20     # events until the entity's own baseline is fully trusted
TYPE_MIN_HISTORY = 30        # events until the entity-type baseline is considered reliable
ENTITY_SOURCE_THRESHOLD = 0.7  # weight_entity above this -> baseline_source == "entity"


def _circular_hour_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2) % 24
    return min(diff, 24 - diff)


# ---------------------------------------------------------------------------
# Full per-entity running state (devices / locations / resources / commands /
# auth / time windows) — used for entity-only signals with no sensible
# cross-entity fallback.
# ---------------------------------------------------------------------------
class EntityRunningState:
    def __init__(self):
        self.n_events = 0
        self.last_timestamp = None
        self.last_geo_location = None

        self.seen_devices = set()
        self.device_counts = Counter()
        self.seen_locations = set()
        self.location_counts = Counter()
        self.auth_method_counts = Counter()
        self.resource_counts = Counter()
        self.command_vocab = set()
        self.command_counts = Counter()

        self.n_auth_fail = 0

        self.recent_events = deque()      # (timestamp, auth_success) for window counts
        self.recent_failures = deque()    # timestamps of failed auth
        self.recent_session_durations = deque(maxlen=20)

    def most_common_device_share(self):
        if self.n_events == 0 or not self.device_counts:
            return 0.0
        return self.device_counts.most_common(1)[0][1] / self.n_events

    def auth_method_frequency(self, method):
        if self.n_events == 0:
            return 0.0
        return self.auth_method_counts.get(method, 0) / self.n_events

    def command_frequency(self, cmd):
        if self.n_events == 0:
            return 0.0
        return self.command_counts.get(cmd, 0) / self.n_events

    def purge_windows(self, current_ts):
        import pandas as pd
        cutoff_1h = current_ts - pd.Timedelta(hours=1)
        cutoff_10m = current_ts - pd.Timedelta(minutes=10)
        while self.recent_events and self.recent_events[0][0] < cutoff_1h:
            self.recent_events.popleft()
        while self.recent_failures and self.recent_failures[0] < cutoff_10m:
            self.recent_failures.popleft()

    def events_last_1h(self, current_ts):
        import pandas as pd
        return sum(1 for ts, _ in self.recent_events if ts >= current_ts - pd.Timedelta(hours=1))

    def events_last_10m(self, current_ts):
        import pandas as pd
        return sum(1 for ts, _ in self.recent_events if ts >= current_ts - pd.Timedelta(minutes=10))

    def failed_attempts_last_10m(self, current_ts):
        import pandas as pd
        return sum(1 for ts in self.recent_failures if ts >= current_ts - pd.Timedelta(minutes=10))

    def update(self, ts, geo_location, device, auth_method, auth_success, resource, commands):
        self.purge_windows(ts)
        self.recent_events.append((ts, auth_success))
        if not auth_success:
            self.recent_failures.append(ts)

        self.seen_devices.add(device)
        self.device_counts[device] += 1
        self.seen_locations.add(geo_location)
        self.location_counts[geo_location] += 1
        self.auth_method_counts[auth_method] += 1
        self.resource_counts[resource] += 1
        for c in commands:
            self.command_vocab.add(c)
            self.command_counts[c] += 1
        if not auth_success:
            self.n_auth_fail += 1

        self.last_timestamp = ts
        self.last_geo_location = geo_location
        self.n_events += 1

    # ---- human-readable summary (for dashboard / SOC copilot) ------------
    def summarize(self, top_k=3):
        top_devices = [d for d, _ in self.device_counts.most_common(top_k)]
        top_locations = [l for l, _ in self.location_counts.most_common(top_k)]
        top_resources = [r for r, _ in self.resource_counts.most_common(top_k)]
        return {
            "total_events": self.n_events,
            "known_devices": top_devices,
            "typical_locations": top_locations,
            "typical_resources": top_resources,
            "auth_failure_rate": (self.n_auth_fail / self.n_events) if self.n_events else 0.0,
        }


# ---------------------------------------------------------------------------
# Lightweight per-level state used ONLY for hierarchical blending
# (session duration, typical hour, resource frequency).
# ---------------------------------------------------------------------------
class _LevelState:
    def __init__(self):
        self.n = 0
        self.session_sum = 0.0
        self.session_sumsq = 0.0
        self.hour_sin_sum = 0.0
        self.hour_cos_sum = 0.0
        self.resource_counts = Counter()

    def mean_std(self):
        if self.n == 0:
            return None, None
        mean = self.session_sum / self.n
        if self.n < 2:
            return mean, None
        var = max(0.0, (self.session_sumsq / self.n) - mean ** 2)
        return mean, var ** 0.5

    def typical_hour(self):
        if self.n == 0:
            return None
        s, c = self.hour_sin_sum / self.n, self.hour_cos_sum / self.n
        angle = math.atan2(s, c)
        return (angle / (2 * math.pi) * 24) % 24

    def resource_freq(self, resource):
        if self.n == 0:
            return None
        return self.resource_counts.get(resource, 0) / self.n

    def update(self, hour, session_duration, resource):
        self.n += 1
        self.session_sum += session_duration
        self.session_sumsq += session_duration ** 2
        angle = (hour / 24) * 2 * math.pi
        self.hour_sin_sum += math.sin(angle)
        self.hour_cos_sum += math.cos(angle)
        self.resource_counts[resource] += 1


class EntityProfiler:
    """Hierarchical entity -> entity_type -> global baseline blender.

    All reads (via `snapshot`) reflect only events processed by `update`
    so far -- callers must call `snapshot` BEFORE `update` for the same
    event to preserve causality.
    """

    def __init__(self):
        self.entity_states = defaultdict(_LevelState)
        self.type_states = defaultdict(_LevelState)
        self.global_state = _LevelState()

    def snapshot(self, entity_id: str, entity_type: str, resource: str, current_hour: float) -> dict:
        e = self.entity_states[entity_id]
        t = self.type_states[entity_type]
        g = self.global_state

        weight_entity = min(1.0, e.n / ENTITY_CONFIDENCE_K)

        if t.n >= TYPE_MIN_HISTORY:
            fb, fallback_label = t, "entity_type"
        else:
            fb, fallback_label = g, "global"

        # --- session duration mean/std ---
        e_mean, e_std = e.mean_std()
        fb_mean, fb_std = fb.mean_std()
        if e_mean is None and fb_mean is None:
            blended_mean, blended_std = 0.0, None
        elif e_mean is None:
            blended_mean, blended_std = fb_mean, fb_std
        elif fb_mean is None:
            blended_mean, blended_std = e_mean, e_std
        else:
            blended_mean = weight_entity * e_mean + (1 - weight_entity) * fb_mean
            e_std_val = e_std if e_std is not None else 0.0
            fb_std_val = fb_std if fb_std is not None else 0.0
            blended_std = weight_entity * e_std_val + (1 - weight_entity) * fb_std_val
            if blended_std < 1e-6:
                blended_std = fb_std_val if fb_std_val > 1e-6 else None

        # --- typical hour (circular blend) ---
        e_hour, fb_hour = e.typical_hour(), fb.typical_hour()
        if e_hour is None:
            blended_hour = fb_hour
        elif fb_hour is None:
            blended_hour = e_hour
        else:
            ex, ey = math.cos(e_hour / 24 * 2 * math.pi), math.sin(e_hour / 24 * 2 * math.pi)
            fx, fy = math.cos(fb_hour / 24 * 2 * math.pi), math.sin(fb_hour / 24 * 2 * math.pi)
            bx = weight_entity * ex + (1 - weight_entity) * fx
            by = weight_entity * ey + (1 - weight_entity) * fy
            blended_hour = (math.atan2(by, bx) / (2 * math.pi) * 24) % 24

        # --- resource access frequency ---
        e_res_freq, fb_res_freq = e.resource_freq(resource), fb.resource_freq(resource)
        if e_res_freq is None and fb_res_freq is None:
            blended_res_freq = 0.0
        elif e_res_freq is None:
            blended_res_freq = fb_res_freq
        elif fb_res_freq is None:
            blended_res_freq = e_res_freq
        else:
            blended_res_freq = weight_entity * e_res_freq + (1 - weight_entity) * fb_res_freq

        if weight_entity >= ENTITY_SOURCE_THRESHOLD:
            baseline_source = "entity"
        else:
            baseline_source = fallback_label

        hour_deviation = (
            _circular_hour_distance(current_hour, blended_hour) / 12.0 if blended_hour is not None else 0.0
        )

        return {
            "session_mean": blended_mean,
            "session_std": blended_std,
            "typical_hour": blended_hour,
            "hour_deviation_norm": hour_deviation,
            "resource_freq": blended_res_freq,
            "weight_entity": weight_entity,
            "baseline_source": baseline_source,
            "entity_history_count": e.n,
        }

    def update(self, entity_id: str, entity_type: str, resource: str, hour: float, session_duration: float):
        self.entity_states[entity_id].update(hour, session_duration, resource)
        self.type_states[entity_type].update(hour, session_duration, resource)
        self.global_state.update(hour, session_duration, resource)
