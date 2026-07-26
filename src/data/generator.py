"""
SentinelAI — Synthetic cybersecurity telemetry generator.

Produces:
    1. Raw events (data/raw/events.csv)        -> NO ground-truth columns
    2. Ground truth labels (data/ground_truth/labels.csv)
    3. Entity profiles (data/processed/entity_profiles.json) -> the
       "true" generative behavioral parameters, useful for debugging /
       judge Q&A ("how do we know the baseline is realistic?"), but
       NEVER fed into feature engineering as-is (features must be
       *learned* from event history, not read from this file).

Design notes
------------
* Deterministic: every run with the same RANDOM_SEED produces the same
  dataset.
* Ground truth is generated in a completely separate structure and
  joined only for evaluation, never for training features.
* Attacks are injected as additional, explicitly-labeled events mixed
  into each entity's normal timeline — they are not a separate,
  trivially-filterable block at the end of the file.
"""

import json
import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel_ai.generator")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def haversine_km(loc_a: str, loc_b: str) -> float:
    """Great-circle distance in km between two named GEO_LOCATIONS entries."""
    lat1, lon1 = cfg.GEO_LOCATIONS[loc_a]
    lat2, lon2 = cfg.GEO_LOCATIONS[loc_b]
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def weighted_choice(rng: random.Random, options: List, weights: List[float]):
    return rng.choices(options, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Entity profile
# ---------------------------------------------------------------------------
@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str
    home_locations: List[str]
    location_weights: List[float]
    known_devices: List[str]
    device_weights: List[float]
    ip_subnet: str                     # e.g. "10.42.7"
    active_hour_start: int
    active_hour_end: int
    active_on_weekends: bool
    typical_auth_method: str
    resource_weights: Dict[str, float]
    session_mean_minutes: float
    session_std_minutes: float
    command_pool: List[str]
    command_weights: List[float]
    activity_level: float               # relative event volume multiplier

    def sample_ip(self, rng: random.Random) -> str:
        return f"{self.ip_subnet}.{rng.randint(2, 254)}"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
class SyntheticDataGenerator:
    def __init__(self, seed: int = cfg.RANDOM_SEED):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.sim_start = datetime.strptime(cfg.SIM_START, "%Y-%m-%d %H:%M:%S")
        self.sim_end = self.sim_start + timedelta(days=cfg.SIMULATION_DAYS)
        self.entities: Dict[str, EntityProfile] = {}
        self._event_counter = 0

    # ---- ID helpers -----------------------------------------------------
    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"EVT-{self._event_counter:07d}"

    # ---- Stage: entity construction -------------------------------------
    def build_entities(self) -> None:
        logger.info("Building entity behavioral profiles...")
        all_resources = list(cfg.RESOURCES.keys())
        all_locations = list(cfg.GEO_LOCATIONS.keys())
        # Reserve a subset of "common/home" locations so travel-anomaly
        # destinations remain genuinely unusual for most entities.
        # GEO_LOCATIONS order: Chennai..Singapore (7 APAC/home cities), then
        # Frankfurt/London/New_York/Moscow/Lagos (5 rare/travel cities).
        home_pool = all_locations[:7]
        rare_pool = all_locations[7:]

        entity_index = 0
        for i in range(cfg.NUM_USER_ENTITIES):
            entity_index += 1
            entity_id = f"USER_{entity_index:03d}"
            n_home = self.rng.choice([1, 1, 1, 2])  # most people have 1 home city
            homes = self.rng.sample(home_pool, k=n_home)
            home_weights = sorted([self.rng.uniform(0.5, 1.0) for _ in homes], reverse=True)
            home_weights = [w / sum(home_weights) for w in home_weights]

            n_devices = self.rng.choice([1, 2, 2, 3])
            devices = [f"DEV_{self.rng.randint(100, 999)}" for _ in range(n_devices)]
            dev_weights = sorted([self.rng.uniform(0.4, 1.0) for _ in devices], reverse=True)
            dev_weights = [w / sum(dev_weights) for w in dev_weights]

            subnet = f"10.{self.rng.randint(0, 40)}.{self.rng.randint(0, 254)}"

            start_hour = self.rng.choice([6, 7, 8, 9])
            end_hour = start_hour + self.rng.choice([8, 9, 10])

            n_res = self.rng.randint(2, 4)
            chosen_resources = self.rng.sample(
                [r for r, s in cfg.RESOURCES.items() if s != "high"] +
                self.rng.sample([r for r, s in cfg.RESOURCES.items() if s == "high"], 1),
                k=min(n_res, len(all_resources)),
            )
            res_weights_raw = sorted([self.rng.uniform(0.3, 1.0) for _ in chosen_resources], reverse=True)
            res_weights = {r: w / sum(res_weights_raw) for r, w in zip(chosen_resources, res_weights_raw)}

            n_cmds = self.rng.randint(4, 7)
            cmds = self.rng.sample(cfg.COMMAND_POOL, k=n_cmds)
            cmd_weights = [self.rng.uniform(0.3, 1.0) for _ in cmds]
            cmd_weights = [w / sum(cmd_weights) for w in cmd_weights]

            profile = EntityProfile(
                entity_id=entity_id,
                entity_type="human_user",
                home_locations=homes,
                location_weights=home_weights,
                known_devices=devices,
                device_weights=dev_weights,
                ip_subnet=subnet,
                active_hour_start=start_hour,
                active_hour_end=min(end_hour, 22),
                active_on_weekends=self.rng.random() < 0.12,
                typical_auth_method=self.rng.choice(cfg.AUTH_METHODS[:4]),  # humans rarely use api_key primarily
                resource_weights=res_weights,
                session_mean_minutes=self.rng.uniform(15, 45),
                session_std_minutes=self.rng.uniform(4, 10),
                command_pool=cmds,
                command_weights=cmd_weights,
                activity_level=self.rng.uniform(0.6, 1.6),
            )
            self.entities[entity_id] = profile

        for i in range(cfg.NUM_SERVICE_ENTITIES):
            entity_index += 1
            entity_id = f"SVC_{entity_index:03d}"
            homes = self.rng.sample(home_pool, k=1)
            subnet = f"10.{self.rng.randint(0, 40)}.{self.rng.randint(0, 254)}"
            devices = [f"DEV_{self.rng.randint(100, 999)}"]  # service accounts run on one host
            n_res = self.rng.randint(1, 2)
            chosen_resources = self.rng.sample(all_resources, k=n_res)
            res_weights_raw = sorted([self.rng.uniform(0.5, 1.0) for _ in chosen_resources], reverse=True)
            res_weights = {r: w / sum(res_weights_raw) for r, w in zip(chosen_resources, res_weights_raw)}
            cmds = self.rng.sample(cfg.COMMAND_POOL, k=3)
            cmd_weights = [1 / 3, 1 / 3, 1 / 3]

            profile = EntityProfile(
                entity_id=entity_id,
                entity_type="service_account",
                home_locations=homes,
                location_weights=[1.0],
                known_devices=devices,
                device_weights=[1.0],
                ip_subnet=subnet,
                active_hour_start=0,
                active_hour_end=23,
                active_on_weekends=True,
                typical_auth_method="api_key",
                resource_weights=res_weights,
                session_mean_minutes=self.rng.uniform(1, 5),
                session_std_minutes=self.rng.uniform(0.2, 1.0),
                command_pool=cmds,
                command_weights=cmd_weights,
                activity_level=self.rng.uniform(0.8, 1.2),
            )
            self.entities[entity_id] = profile

        # Edge/IoT-OT devices: a third, distinctly-behaved entity type per the
        # assessment's domain-agnostic framing (industrial edge gateway, POS
        # terminal, home IoT hub). Distinguishing characteristics vs. the
        # human/service entities above:
        #   - device_fingerprint is an OS/firmware + MAC-address style string
        #     (matches the assessment schema's exact wording), not an opaque
        #     DEV_xxx ID -- and there is exactly ONE per entity, never rotated
        #     under normal operation, making device_spoofing maximally
        #     meaningful for this entity type (ANY change is anomalous).
        #   - certificate-based auth almost exclusively (machine identity,
        #     not a human password).
        #   - near-24/7, low-variance, heartbeat-like timing -- no lunch
        #     breaks, no weekends, minimal session-length variability.
        #   - resource access is device-function endpoints only
        #     (/telemetry, /sensor-data, /firmware-update), never HR/finance.
        #   - fixed to a single physical location (a gateway doesn't travel).
        OS_FIRMWARE_VERSIONS = ["LinuxOT-4.2", "EdgeOS-2.1", "IoTGate-7.0", "SecureRTOS-3.4"]
        edge_resources = [r for r in ["/telemetry", "/sensor-data", "/firmware-update"] if r in cfg.RESOURCES]

        for i in range(cfg.NUM_EDGE_DEVICE_ENTITIES):
            entity_index += 1
            entity_id = f"EDGE_{entity_index:03d}"
            home = self.rng.sample(home_pool, k=1)
            subnet = f"192.168.{self.rng.randint(0, 254)}"  # distinct private range from human/service 10.x.x

            mac = ":".join(f"{self.rng.randint(0, 255):02X}" for _ in range(6))
            os_version = self.rng.choice(OS_FIRMWARE_VERSIONS)
            device_fingerprint = f"{os_version}/{mac}"

            n_res = min(len(edge_resources), self.rng.randint(2, 3))
            chosen_resources = self.rng.sample(edge_resources, k=n_res)
            res_weights_raw = sorted([self.rng.uniform(0.4, 1.0) for _ in chosen_resources], reverse=True)
            res_weights = {r: w / sum(res_weights_raw) for r, w in zip(chosen_resources, res_weights_raw)}
            cmds = self.rng.sample(cfg.COMMAND_POOL, k=2)
            cmd_weights = [0.6, 0.4]

            profile = EntityProfile(
                entity_id=entity_id,
                entity_type="edge_device",
                home_locations=home,
                location_weights=[1.0],
                known_devices=[device_fingerprint],
                device_weights=[1.0],
                ip_subnet=subnet,
                active_hour_start=0,
                active_hour_end=23,
                active_on_weekends=True,
                typical_auth_method="certificate",
                resource_weights=res_weights,
                session_mean_minutes=self.rng.uniform(0.2, 0.8),   # brief machine-to-machine calls
                session_std_minutes=self.rng.uniform(0.03, 0.1),   # very low variance -- heartbeat-like
                command_pool=cmds,
                command_weights=cmd_weights,
                activity_level=self.rng.uniform(1.2, 2.2),         # frequent check-ins
            )
            self.entities[entity_id] = profile

        logger.info(f"Built {len(self.entities)} entity profiles "
                    f"({cfg.NUM_USER_ENTITIES} human, {cfg.NUM_SERVICE_ENTITIES} service, "
                    f"{cfg.NUM_EDGE_DEVICE_ENTITIES} edge_device).")

    # ---- Stage: normal event generation ----------------------------------
    def _sample_active_timestamp(self, profile: EntityProfile) -> datetime:
        """Pick a timestamp weighted toward the entity's active hours/days."""
        for _ in range(50):  # rejection sampling, bounded
            day_offset = self.rng.uniform(0, cfg.SIMULATION_DAYS)
            ts = self.sim_start + timedelta(days=day_offset)
            if ts.weekday() >= 5 and not profile.active_on_weekends:
                if self.rng.random() > 0.05:  # small chance of an off-day login anyway
                    continue
            # bias hour toward active window using a triangular distribution
            center = (profile.active_hour_start + profile.active_hour_end) / 2
            hour = self.rng.triangular(
                max(0, profile.active_hour_start - 2),
                min(23, profile.active_hour_end + 2),
                center,
            )
            minute = self.rng.uniform(0, 59)
            ts = ts.replace(hour=int(hour) % 24, minute=int(minute), second=self.rng.randint(0, 59))
            return ts
        return self.sim_start + timedelta(days=self.rng.uniform(0, cfg.SIMULATION_DAYS))

    def _make_normal_event(self, profile: EntityProfile, ts: datetime) -> dict:
        device = weighted_choice(self.rng, profile.known_devices, profile.device_weights)
        location = weighted_choice(self.rng, profile.home_locations, profile.location_weights)
        ip = profile.sample_ip(self.rng)
        resource = weighted_choice(self.rng, list(profile.resource_weights.keys()),
                                    list(profile.resource_weights.values()))
        auth_method = profile.typical_auth_method if self.rng.random() > 0.08 else self.rng.choice(cfg.AUTH_METHODS)
        auth_success = self.rng.random() > 0.03  # occasional honest mistype
        session_duration = max(0.5, self.np_rng.normal(profile.session_mean_minutes, profile.session_std_minutes))
        n_cmds = self.rng.randint(2, 6)
        commands = self.rng.choices(profile.command_pool, weights=profile.command_weights, k=n_cmds)

        return {
            "event_id": self._next_event_id(),
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": ip,
            "geo_location": location,
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_success": bool(auth_success),
            "session_duration": round(float(session_duration), 2),
            "command_sequence": "|".join(commands),
            "device_fingerprint": device,
        }

    def generate_normal_events(self, n_events: int) -> List[dict]:
        logger.info(f"Generating {n_events} normal events...")
        weights = [p.activity_level for p in self.entities.values()]
        entity_list = list(self.entities.values())
        events = []
        for _ in range(n_events):
            profile = weighted_choice(self.rng, entity_list, weights)
            ts = self._sample_active_timestamp(profile)
            events.append(self._make_normal_event(profile, ts))
        return events

    # ---- Attack injections -------------------------------------------------
    def _label(self, event: dict, attack_type: str) -> dict:
        event["_is_anomaly"] = 1
        event["_attack_type"] = attack_type
        return event

    def _stratified_start_day_offset(self, incident_index: int) -> float:
        """Round-robin assigns incidents across the train/val/test day-ranges
        (proportional to TRAIN_FRAC/VAL_FRAC) so BURST-STYLE attacks (all
        events within minutes of each other) don't miss an entire split
        purely by chance. With only ~6-10 independent incidents per attack
        type, pure-uniform placement across 30 days occasionally misses the
        6-day test region entirely (observed empirically) -- this guarantees
        coverage whenever an attack type has >= 3 incidents, which all of
        ours do."""
        position = incident_index % 3
        if position == 0:
            lo, hi = 0.0, cfg.TRAIN_FRAC
        elif position == 1:
            lo, hi = cfg.TRAIN_FRAC, cfg.TRAIN_FRAC + cfg.VAL_FRAC
        else:
            lo, hi = cfg.TRAIN_FRAC + cfg.VAL_FRAC, 1.0
        return self.rng.uniform(lo * cfg.SIMULATION_DAYS, hi * cfg.SIMULATION_DAYS)

    def inject_brute_force(self, n_events_budget: int) -> List[dict]:
        """Many rapid failed logins against one entity from one attacker IP."""
        events = []
        entity_list = list(self.entities.values())
        incident_index = 0
        while len(events) < n_events_budget:
            profile = self.rng.choice(entity_list)
            burst_len = self.rng.randint(6, 14)
            attacker_ip = f"185.{self.rng.randint(0, 254)}.{self.rng.randint(0, 254)}.{self.rng.randint(1, 254)}"
            start_ts = self.sim_start + timedelta(days=self._stratified_start_day_offset(incident_index))
            incident_index += 1
            for i in range(burst_len):
                ts = start_ts + timedelta(seconds=self.rng.randint(5, 40) * (i + 1))
                success = (i == burst_len - 1) and self.rng.random() < 0.3  # rare final success
                ev = {
                    "event_id": self._next_event_id(),
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": ts,
                    "source_ip": attacker_ip,
                    "geo_location": self.rng.choice(list(cfg.GEO_LOCATIONS.keys())),
                    "resource_accessed": self.rng.choice(list(cfg.RESOURCES.keys())),
                    "auth_method": "password",
                    "auth_success": bool(success),
                    "session_duration": round(self.rng.uniform(0.1, 1.0), 2),
                    "command_sequence": "",
                    "device_fingerprint": f"DEV_{self.rng.randint(100, 999)}",
                }
                events.append(self._label(ev, "brute_force"))
                if len(events) >= n_events_budget:
                    break
        return events[:n_events_budget]

    def inject_impossible_travel(self, n_events_budget: int) -> List[dict]:
        """A login far from the entity's home location shortly after a normal login."""
        events = []
        entity_list = list(self.entities.values())
        rare_locations = list(cfg.GEO_LOCATIONS.keys())[7:]  # far, uncommon cities
        while len(events) < n_events_budget:
            profile = self.rng.choice(entity_list)
            base_ts = self._sample_active_timestamp(profile)
            home = profile.home_locations[0]
            far_location = self.rng.choice([l for l in rare_locations if l != home] or rare_locations)
            distance = haversine_km(home, far_location)
            # impossible speed: gap small enough that required velocity > commercial flight speed
            max_minutes_for_impossible = max(5, (distance / 900) * 60 * self.rng.uniform(0.2, 0.6))
            gap_minutes = self.rng.uniform(5, max(6, max_minutes_for_impossible))
            ts = base_ts + timedelta(minutes=gap_minutes)
            ev = {
                "event_id": self._next_event_id(),
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": ts,
                "source_ip": f"{self.rng.randint(20, 200)}.{self.rng.randint(0, 254)}.{self.rng.randint(0, 254)}.{self.rng.randint(1, 254)}",
                "geo_location": far_location,
                "resource_accessed": self.rng.choice(list(profile.resource_weights.keys())),
                "auth_method": profile.typical_auth_method,
                "auth_success": True,
                "session_duration": round(self.rng.uniform(5, 30), 2),
                "command_sequence": "|".join(self.rng.choices(profile.command_pool, k=3)),
                "device_fingerprint": self.rng.choice(
                    [profile.known_devices[0], f"DEV_{self.rng.randint(100, 999)}"]
                ),
            }
            events.append(self._label(ev, "impossible_travel"))
        return events[:n_events_budget]

    def inject_credential_stuffing(self, n_events_budget: int) -> List[dict]:
        """One attacker IP probes many distinct entities in a short window."""
        events = []
        entity_list = list(self.entities.values())
        incident_index = 0
        while len(events) < n_events_budget:
            attacker_ip = f"91.{self.rng.randint(0, 254)}.{self.rng.randint(0, 254)}.{self.rng.randint(1, 254)}"
            targets = self.rng.sample(entity_list, k=min(len(entity_list), self.rng.randint(6, 18)))
            start_ts = self.sim_start + timedelta(days=self._stratified_start_day_offset(incident_index))
            incident_index += 1
            for i, profile in enumerate(targets):
                ts = start_ts + timedelta(seconds=self.rng.randint(2, 20) * (i + 1))
                success = self.rng.random() < 0.08
                ev = {
                    "event_id": self._next_event_id(),
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": ts,
                    "source_ip": attacker_ip,
                    "geo_location": self.rng.choice(list(cfg.GEO_LOCATIONS.keys())),
                    "resource_accessed": self.rng.choice(list(cfg.RESOURCES.keys())),
                    "auth_method": "password",
                    "auth_success": bool(success),
                    "session_duration": round(self.rng.uniform(0.1, 1.0), 2),
                    "command_sequence": "",
                    "device_fingerprint": f"DEV_{self.rng.randint(100, 999)}",
                }
                events.append(self._label(ev, "credential_stuffing"))
                if len(events) >= n_events_budget:
                    break
        return events[:n_events_budget]

    def inject_lateral_movement(self, n_events_budget: int) -> List[dict]:
        """Rapid-fire access to multiple unfamiliar (often sensitive) resources."""
        events = []
        entity_list = list(self.entities.values())
        sensitive_resources = [r for r, s in cfg.RESOURCES.items() if s in ("medium", "high")]
        incident_index = 0
        while len(events) < n_events_budget:
            profile = self.rng.choice(entity_list)
            start_ts = self.sim_start + timedelta(days=self._stratified_start_day_offset(incident_index))
            incident_index += 1
            chain_len = self.rng.randint(3, 6)
            spoof_device = f"DEV_{self.rng.randint(100, 999)}"
            unfamiliar = [r for r in sensitive_resources if r not in profile.resource_weights]
            hop_resources = self.rng.sample(unfamiliar or sensitive_resources, k=min(chain_len, len(unfamiliar or sensitive_resources)))
            for i, res in enumerate(hop_resources):
                ts = start_ts + timedelta(minutes=self.rng.uniform(1, 4) * (i + 1))
                ev = {
                    "event_id": self._next_event_id(),
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": ts,
                    "source_ip": f"{self.rng.randint(20, 200)}.{self.rng.randint(0, 254)}.{self.rng.randint(0, 254)}.{self.rng.randint(1, 254)}",
                    "geo_location": self.rng.choice(list(cfg.GEO_LOCATIONS.keys())),
                    "resource_accessed": res,
                    "auth_method": profile.typical_auth_method,
                    "auth_success": True,
                    "session_duration": round(self.rng.uniform(1, 8), 2),
                    "command_sequence": "|".join(self.rng.sample(cfg.COMMAND_POOL, k=3)),
                    "device_fingerprint": spoof_device,
                }
                events.append(self._label(ev, "lateral_movement"))
                if len(events) >= n_events_budget:
                    break
        return events[:n_events_budget]

    def inject_device_spoofing(self, n_events_budget: int) -> List[dict]:
        """A never-before-seen device fingerprint is suddenly used; sometimes
        the same spoofed fingerprint reappears across multiple entities."""
        events = []
        entity_list = list(self.entities.values())
        shared_spoof_pool = [f"SPOOF_{self.rng.randint(1000, 9999)}" for _ in range(4)]
        while len(events) < n_events_budget:
            profile = self.rng.choice(entity_list)
            ts = self._sample_active_timestamp(profile)
            device = self.rng.choice(shared_spoof_pool)
            ev = {
                "event_id": self._next_event_id(),
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": ts,
                "source_ip": f"{self.rng.randint(20, 200)}.{self.rng.randint(0, 254)}.{self.rng.randint(0, 254)}.{self.rng.randint(1, 254)}",
                "geo_location": self.rng.choice(profile.home_locations + [self.rng.choice(list(cfg.GEO_LOCATIONS.keys()))]),
                "resource_accessed": weighted_choice(self.rng, list(profile.resource_weights.keys()),
                                                      list(profile.resource_weights.values())),
                "auth_method": profile.typical_auth_method,
                "auth_success": True,
                "session_duration": round(self.rng.uniform(2, 20), 2),
                "command_sequence": "|".join(self.rng.choices(profile.command_pool, k=3)),
                "device_fingerprint": device,
            }
            events.append(self._label(ev, "device_spoofing"))
        return events[:n_events_budget]

    def inject_low_slow_exfiltration(self, n_events_budget: int) -> List[dict]:
        """Sparse, spread-out access to sensitive resources with export-like commands."""
        events = []
        entity_list = list(self.entities.values())
        high_sens = [r for r, s in cfg.RESOURCES.items() if s == "high"]
        exfil_commands = ["export_csv", "zip_archive", "scp", "curl"]
        while len(events) < n_events_budget:
            profile = self.rng.choice(entity_list)
            incident_len = self.rng.randint(5, 10)
            day_positions = sorted(self.rng.sample(range(cfg.SIMULATION_DAYS), k=min(incident_len, cfg.SIMULATION_DAYS)))
            for day in day_positions:
                ts = self.sim_start + timedelta(days=day, hours=self.rng.uniform(profile.active_hour_start,
                                                                                   profile.active_hour_end))
                ev = {
                    "event_id": self._next_event_id(),
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": ts,
                    "source_ip": profile.sample_ip(self.rng),
                    "geo_location": self.rng.choice(profile.home_locations),
                    "resource_accessed": self.rng.choice(high_sens),
                    "auth_method": profile.typical_auth_method,
                    "auth_success": True,
                    "session_duration": round(self.rng.uniform(20, 60), 2),  # longer, quiet sessions
                    "command_sequence": "|".join(self.rng.choices(exfil_commands, k=3)),
                    "device_fingerprint": self.rng.choice(profile.known_devices),
                }
                events.append(self._label(ev, "low_slow_exfiltration"))
                if len(events) >= n_events_budget:
                    break
        return events[:n_events_budget]

    def inject_insider_drift(self, n_events_budget: int) -> List[dict]:
        """Gradual behavioral drift: only the later, more-deviated portion of
        the drift window is labeled anomalous, simulating a slow insider
        threat that starts subtly and becomes clearer over time."""
        events = []
        entity_list = list(self.entities.values())
        anomalous_so_far = 0
        while anomalous_so_far < n_events_budget:
            profile = self.rng.choice(entity_list)
            drift_len = self.rng.randint(10, 20)
            # Spread incidents across most of the simulation window (not just
            # a narrow mid-window band) so labeled insider_drift events land
            # in the train, validation, AND test chronological splits --
            # otherwise a classifier can end up with zero training examples
            # of this class purely from unlucky timing.
            drift_start_day = self.rng.uniform(2, max(3, cfg.SIMULATION_DAYS - 8))
            new_device = f"DEV_{self.rng.randint(100, 999)}"
            drift_resource_pool = [r for r, s in cfg.RESOURCES.items() if s == "high"]
            for i in range(drift_len):
                progress = i / max(1, drift_len - 1)  # 0 -> 1 across the window
                ts = self.sim_start + timedelta(days=drift_start_day + i * 0.8,
                                                 hours=profile.active_hour_end - 1 + progress * 4)
                use_new_device = self.rng.random() < progress  # more likely late in the window
                use_sensitive = self.rng.random() < progress
                ev = {
                    "event_id": self._next_event_id(),
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": ts,
                    "source_ip": profile.sample_ip(self.rng),
                    "geo_location": self.rng.choice(profile.home_locations),
                    "resource_accessed": (self.rng.choice(drift_resource_pool) if use_sensitive
                                          else weighted_choice(self.rng, list(profile.resource_weights.keys()),
                                                                list(profile.resource_weights.values()))),
                    "auth_method": profile.typical_auth_method,
                    "auth_success": True,
                    "session_duration": round(self.np_rng.normal(
                        profile.session_mean_minutes * (1 + 0.5 * progress), profile.session_std_minutes), 2),
                    "command_sequence": "|".join(self.rng.choices(profile.command_pool, k=3)),
                    "device_fingerprint": new_device if use_new_device else self.rng.choice(profile.known_devices),
                }
                # Only the later half of the drift window (progress >= 0.5) counts
                # as ground-truth anomalous — the earlier half is genuinely
                # ambiguous/normal-looking, by design.
                if progress >= 0.5:
                    events.append(self._label(ev, "insider_drift"))
                    anomalous_so_far += 1
                else:
                    ev["_is_anomaly"] = 0
                    ev["_attack_type"] = "normal"
                    events.append(ev)
                if anomalous_so_far >= n_events_budget:
                    break
        # Keep all generated (including the ambiguous normal-labeled half for realism),
        # but cap the anomalous portion at budget.
        anomalous = [e for e in events if e["_is_anomaly"] == 1][:n_events_budget]
        normal_half = [e for e in events if e["_is_anomaly"] == 0]
        return anomalous + normal_half

    # ---- Orchestration -----------------------------------------------------
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.build_entities()

        anomaly_count = int(round(cfg.NUM_EVENTS * cfg.ANOMALY_RATE))
        normal_count = cfg.NUM_EVENTS - anomaly_count

        normal_events = self.generate_normal_events(normal_count)
        for e in normal_events:
            e["_is_anomaly"] = 0
            e["_attack_type"] = "normal"

        logger.info(f"Injecting attacks (target ~{anomaly_count} anomalous events)...")
        attack_events: List[dict] = []
        injectors = {
            "brute_force": self.inject_brute_force,
            "impossible_travel": self.inject_impossible_travel,
            "credential_stuffing": self.inject_credential_stuffing,
            "lateral_movement": self.inject_lateral_movement,
            "device_spoofing": self.inject_device_spoofing,
            "low_slow_exfiltration": self.inject_low_slow_exfiltration,
            "insider_drift": self.inject_insider_drift,
        }
        for attack_type, weight in cfg.ATTACK_WEIGHTS.items():
            budget = max(1, int(round(anomaly_count * weight)))
            events = injectors[attack_type](budget)
            n_labeled = sum(1 for e in events if e["_is_anomaly"] == 1)
            logger.info(f"  {attack_type:<22s} budget={budget:<5d} generated={len(events):<5d} labeled_anomalous={n_labeled}")
            attack_events.extend(events)

        all_events = normal_events + attack_events
        self.rng.shuffle(all_events)  # break any artificial ordering before time-sort

        df = pd.DataFrame(all_events)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Re-assign event_id in chronological order for clean downstream indexing
        df["event_id"] = [f"EVT-{i+1:07d}" for i in range(len(df))]

        ground_truth = df[["event_id", "_is_anomaly", "_attack_type"]].rename(
            columns={"_is_anomaly": "is_anomaly", "_attack_type": "attack_type"}
        )
        raw_events = df.drop(columns=["_is_anomaly", "_attack_type"])

        return raw_events, ground_truth

    def export_entity_profiles(self, path: str) -> None:
        serializable = {}
        for eid, p in self.entities.items():
            serializable[eid] = {
                "entity_type": p.entity_type,
                "home_locations": p.home_locations,
                "location_weights": p.location_weights,
                "known_devices": p.known_devices,
                "device_weights": p.device_weights,
                "ip_subnet": p.ip_subnet,
                "active_hour_start": p.active_hour_start,
                "active_hour_end": p.active_hour_end,
                "active_on_weekends": p.active_on_weekends,
                "typical_auth_method": p.typical_auth_method,
                "resource_weights": p.resource_weights,
                "session_mean_minutes": p.session_mean_minutes,
                "session_std_minutes": p.session_std_minutes,
                "command_pool": p.command_pool,
                "command_weights": p.command_weights,
                "activity_level": p.activity_level,
            }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
