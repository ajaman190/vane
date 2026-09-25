"""Editable starter question maps for common decision workflows.

No torch. Callers may mutate the returned dicts; each call builds a fresh copy.
"""

from __future__ import annotations

from typing import Any


def triage_questions() -> dict[str, Any]:
    """Support / intake triage: urgency, department, severity."""
    return {
        "urgent": {
            "type": "noul",
            "instructions": "Is this request urgent and needs immediate attention?",
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "payments, invoices, refunds",
                "technical": "bugs, outages, product defects",
                "sales": "pricing, upgrades, new purchases",
                "other": "none of the above",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the issue for the user?",
            "criteria": ["Low", "Medium", "High", "Critical"],
        },
    }


def guard_questions() -> dict[str, Any]:
    """Safety / policy guardrails before acting on a request."""
    return {
        "safe_to_proceed": {
            "type": "noul",
            "instructions": "Is it safe to proceed with the requested action?",
        },
        "risk_category": {
            "type": "choice",
            "instructions": "Primary risk category if any.",
            "criteria": {
                "none": "no material risk",
                "privacy": "personal or sensitive data exposure",
                "security": "credential, access, or exploit risk",
                "legal": "legal, compliance, or contractual risk",
                "other": "other material risk",
            },
        },
        "caution": {
            "type": "score",
            "instructions": "How much caution should an operator apply?",
            "criteria": ["Proceed", "Review", "Escalate", "Block"],
        },
    }


def moderation_questions() -> dict[str, Any]:
    """Content moderation over user- or model-generated text."""
    return {
        "violates_policy": {
            "type": "noul",
            "instructions": "Does this content violate the moderation policy?",
        },
        "category": {
            "type": "choice",
            "instructions": "Dominant moderation category.",
            "criteria": {
                "ok": "allowed content",
                "hate": "hate or harassment",
                "violence": "violence or self-harm",
                "sexual": "sexual or adult content",
                "spam": "spam or scams",
                "other": "other policy concern",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the policy concern?",
            "criteria": ["None", "Mild", "Moderate", "Severe"],
        },
    }
