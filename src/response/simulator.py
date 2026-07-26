"""
SentinelAI — Response Action Simulator.

CRITICAL SAFETY PRINCIPLE: nothing in this module ever calls a real
system. Every method here is a DRY RUN that constructs the exact API
request that WOULD be sent to a real Identity Provider / firewall /
EDR / ticketing system, and returns it for display -- it is the
analyst's job to review and, if they choose, execute the real action
through their actual tools. This is intentional: security automation
that can autonomously suspend accounts or block IPs without human
approval is a genuine operational and legal risk in a real
organization, and this project does not pretend otherwise.

Every simulated action is appended to an audit log
(reports/response_audit_log.json) recording what was proposed, when,
and that it was a SIMULATION -- demonstrating the audit-trail pattern
a real SOAR integration would need, without the risk of a real one.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings as cfg


def _mock_request(method: str, url: str, payload: dict) -> dict:
    return {"method": method, "url": url, "payload": payload}


class ActionSimulator:
    """Constructs realistic mock API requests for each playbook action
    type. None of these are executed -- see module docstring."""

    def build_request(self, action_type: str, entity_id: str = None, ip: str = None,
                       device_id: str = None, resource: str = None, ticket_priority: str = "P2") -> dict:
        builders = {
            "suspend_account": self._suspend_account,
            "force_password_reset": self._force_password_reset,
            "block_ip": self._block_ip,
            "isolate_device": self._isolate_device,
            "revoke_resource_access": self._revoke_resource_access,
            "notify_manager": self._notify_manager,
            "open_incident_ticket": self._open_incident_ticket,
        }
        builder = builders.get(action_type)
        if builder is None:
            return {"error": f"Unknown action_type '{action_type}'"}
        return builder(entity_id=entity_id, ip=ip, device_id=device_id,
                        resource=resource, ticket_priority=ticket_priority)

    def _suspend_account(self, entity_id, **_):
        return _mock_request("POST", f"https://idp.example.com/api/v1/users/{entity_id}/suspend",
                              {"reason": "SentinelAI automated risk detection", "requires_manual_reactivation": True})

    def _force_password_reset(self, entity_id, **_):
        return _mock_request("POST", f"https://idp.example.com/api/v1/users/{entity_id}/force-password-reset",
                              {"notify_user": True, "invalidate_active_sessions": True})

    def _block_ip(self, ip, **_):
        ip = ip or "0.0.0.0/32"
        return _mock_request("POST", "https://waf.example.com/api/v1/rules/block",
                              {"ip_range": ip, "duration_minutes": 1440, "reason": "SentinelAI automated risk detection"})

    def _isolate_device(self, device_id, **_):
        return _mock_request("POST", f"https://edr.example.com/api/v1/devices/{device_id}/isolate",
                              {"isolation_type": "network", "reason": "SentinelAI lateral movement detection"})

    def _revoke_resource_access(self, entity_id, resource, **_):
        return _mock_request("DELETE", f"https://iam.example.com/api/v1/users/{entity_id}/access/{resource}",
                              {"reason": "SentinelAI low-and-slow exfiltration detection"})

    def _notify_manager(self, entity_id, **_):
        return _mock_request("POST", "https://comms.example.com/api/v1/notify",
                              {"recipient": f"manager_of:{entity_id}", "channel": "email+slack",
                               "template": "security_alert_review_request"})

    def _open_incident_ticket(self, entity_id, ticket_priority, **_):
        return _mock_request("POST", "https://itsm.example.com/api/v1/incidents",
                              {"priority": ticket_priority, "assignee_group": "SOC-Tier2",
                               "reported_by": "SentinelAI", "affected_entity": entity_id})

    def simulate_and_log(self, action_type: str, approved_by: str, **kwargs) -> dict:
        """The only method that should be called from the dashboard/Copilot.
        Requires an explicit `approved_by` (the analyst's name/ID) --
        there is no code path in this project that executes an action
        without a human approval string being provided."""
        request = self.build_request(action_type, **kwargs)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": action_type,
            "approved_by": approved_by,
            "simulated_request": request,
            "status": "SIMULATED_NOT_EXECUTED",
            "note": "This action was NOT sent to any real system. Dry-run only.",
        }
        self._append_audit_log(record)
        return record

    def _append_audit_log(self, record: dict):
        log = []
        if os.path.exists(cfg.RESPONSE_AUDIT_LOG_PATH):
            with open(cfg.RESPONSE_AUDIT_LOG_PATH, "r") as f:
                log = json.load(f)
        log.append(record)
        os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
        with open(cfg.RESPONSE_AUDIT_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)
