"""
SentinelAI — Response Playbook library.

CRITICAL DESIGN PRINCIPLE (same as attack-chain correlation, Stage 9):
this is a DETERMINISTIC lookup table, not an LLM decision. Given an
attack_type and risk_level, the recommended actions are fixed by this
config -- the LLM (Report Agent / Copilot) may explain or narrate a
playbook in natural language, but it never decides WHICH actions to
recommend. That decision stays deterministic and auditable, exactly
like the rest of this system's "ML/rules decide, LLM explains"
philosophy.

Every action is advisory. Nothing in this module ever executes a real
action against a real system -- see src/response/simulator.py for the
dry-run execution layer, which requires explicit analyst approval.

Notably, `insider_drift`'s playbook deliberately recommends "escalate
to manager/HR for review" rather than "suspend account" -- consistent
with Stage 9's concept-drift finding that sustained behavioral change
can reflect a legitimate role change, not just a threat. Automatically
suspending an account over an ambiguous signal would be an irresponsible
default; a human review step is the correct one.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg

# Each action: (action_type, target_description, urgency)
# action_type maps to a simulator method in src/response/simulator.py
PLAYBOOKS = {
    "brute_force": [
        ("block_ip", "Source IP(s) observed in the failed-login burst", "immediate"),
        ("notify_manager", "Entity's manager/security team", "immediate"),
    ],
    "impossible_travel": [
        ("force_password_reset", "The affected entity", "immediate"),
        ("suspend_account", "The affected entity, pending verification", "immediate"),
        ("open_incident_ticket", "SOC incident queue (P1 -- likely account takeover)", "immediate"),
    ],
    "credential_stuffing": [
        ("block_ip", "Source IP(s) observed probing multiple accounts", "immediate"),
        ("notify_manager", "Security team (org-wide exposure -- multiple accounts targeted)", "immediate"),
        ("open_incident_ticket", "SOC incident queue (P2 -- coordinated probing)", "high"),
    ],
    "lateral_movement": [
        ("isolate_device", "The device used during lateral movement", "immediate"),
        ("suspend_account", "The affected entity, pending investigation", "immediate"),
        ("open_incident_ticket", "SOC incident queue (P1 -- active compromise suspected)", "immediate"),
    ],
    "device_spoofing": [
        ("suspend_account", "The affected entity, pending device verification", "immediate"),
        ("block_ip", "Source IP associated with the spoofed device", "high"),
        ("open_incident_ticket", "SOC incident queue (P2)", "high"),
    ],
    "low_slow_exfiltration": [
        ("open_incident_ticket", "SOC incident queue (P1 -- suspected data exfiltration)", "immediate"),
        ("notify_manager", "Data owner / security team", "immediate"),
        ("revoke_resource_access", "Sensitive resource(s) accessed in the pattern", "high"),
    ],
    "insider_drift": [
        # Deliberately NOT suspend_account -- drift can be a legitimate role
        # change. Human review first, per the concept-drift design (Stage 9).
        ("notify_manager", "Entity's manager/HR for a legitimacy check", "review"),
        ("open_incident_ticket", "SOC watchlist (P3 -- monitor, not urgent)", "review"),
    ],
    "normal": [],
}


def get_playbook(attack_type: str, risk_level: str) -> list:
    """Returns the ordered list of recommended actions for this attack
    type. `risk_level` is accepted for future use (e.g. escalating
    urgency further at CRITICAL) but the action SET itself is
    determined by attack_type alone -- kept simple and auditable."""
    actions = PLAYBOOKS.get(attack_type, [])
    return [
        {"action_type": a, "target": target, "urgency": urgency}
        for a, target, urgency in actions
    ]
