"""
SentinelAI — Business Impact Translation.

Bridges the gap between technical risk scores (0-100, meaningless to a
CFO/board) and business language (estimated dollar impact). This is
explicitly ILLUSTRATIVE -- the cost constants in config/settings.py are
placeholders an organization should replace with their own breach-cost
history, cyber-insurance actuarial data, or a benchmark they trust. The
value here is the TRANSLATION MECHANISM, not these specific numbers,
and every output says so explicitly.

Method (simple, transparent, and intentionally NOT a black box):
  1. For each sensitive resource touched by an incident (from the
     attack chain or a single alert), estimate "records at risk" using
     a configurable per-sensitivity-tier heuristic.
  2. Multiply by a configurable cost-per-record.
  3. Model the cost of DELAY: impact grows over time while unaddressed
     (simple, labeled-illustrative exponential growth), so a
     "cost avoided by acting now vs. waiting N hours" figure can be
     shown -- directly motivating fast response.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg


def estimate_records_at_risk(resources_touched: list) -> int:
    """resources_touched: list of resource names (e.g. '/finance-db').
    Looks up each resource's sensitivity tier and sums an illustrative
    records-at-risk estimate per tier."""
    total = 0
    for resource in resources_touched:
        sensitivity = cfg.RESOURCES.get(resource, "low")
        total += cfg.RESPONSE_RECORDS_AT_RISK_BY_SENSITIVITY.get(sensitivity, 0)
    return total


def estimate_business_impact(resources_touched: list, hours_unaddressed: float = 0.0) -> dict:
    """Returns an illustrative dollar-impact estimate and how much it
    grows if left unaddressed for `hours_unaddressed` more hours -- the
    "cost of delay" framing that motivates fast SOC response."""
    records_at_risk = estimate_records_at_risk(resources_touched)
    base_impact = records_at_risk * cfg.RESPONSE_COST_PER_RECORD_USD

    growth_factor = (1 + cfg.RESPONSE_CONTAINMENT_GROWTH_RATE_PER_HOUR) ** hours_unaddressed
    projected_impact = base_impact * growth_factor

    return {
        "records_at_risk_estimate": records_at_risk,
        "immediate_impact_usd": round(base_impact, 2),
        "hours_unaddressed": hours_unaddressed,
        "projected_impact_usd": round(projected_impact, 2),
        "impact_growth_usd": round(projected_impact - base_impact, 2),
        "note": ("ILLUSTRATIVE ESTIMATE using configurable placeholder benchmarks "
                 "(cost/record, records-at-risk-by-sensitivity-tier) in "
                 "config/settings.py -- calibrate to your organization's actual "
                 "breach-cost history or cyber-insurance data before relying on "
                 "these figures operationally."),
    }


def cost_of_delay_curve(resources_touched: list, max_hours: int = 24, step_hours: int = 2) -> list:
    """A curve of projected impact at increasing delay, for a simple
    'cost of waiting' chart in the dashboard."""
    points = []
    h = 0
    while h <= max_hours:
        est = estimate_business_impact(resources_touched, hours_unaddressed=h)
        points.append({"hours_unaddressed": h, "projected_impact_usd": est["projected_impact_usd"]})
        h += step_hours
    return points
