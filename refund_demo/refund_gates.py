# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Refund hard gates for the R2 mechanical regime.

These mirror :func:`mech_gov.governance.primitives.hard_gates.build_default_gates`
in shape (an ordered list of :class:`HardGate`, first match wins) but encode
*refund* policy instead of the banking defaults. They are evaluated BEFORE the
LLM is consulted; when one fires, the decision is mechanical and the model is
never called.

Thresholds come from a ``config`` dict (with sensible defaults), mirroring how
``hard_gates.py`` reads its thresholds.

Note on rationale templates: the reused ``evaluate_hard_gates`` formats each
template with a fixed kwarg set (``risk_score``, ``completeness``, ``amount``,
``n_flags``, ``threshold``, ``min_flags``, ``amount_threshold``, ``decision``).
For these custom gate IDs ``threshold`` is not populated, so the templates only
reference the always-available case kwargs and bake fixed policy thresholds in
as literal text.
"""

from __future__ import annotations

from typing import cast

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.governance.primitives.hard_gates import HardGate
from refund_demo.refund_case import RefundCase

# Default thresholds (overridable via config).
DEFAULT_AMOUNT_LIMIT_USD: float = 2_000.0
DEFAULT_FRAUD_RISK_THRESHOLD: float = 0.7
DEFAULT_NO_EVIDENCE_THRESHOLD: float = 0.15
DEFAULT_ACCOUNT_SWITCH_RISK_THRESHOLD: float = 0.5


def _rc(case: BankingCase) -> RefundCase:
    """Narrow a ``BankingCase`` to ``RefundCase`` for the gate predicates.

    The pipeline always passes a ``RefundCase`` at runtime; the cast keeps the
    predicates type-checkable while matching the ``HardGate.condition`` contract.
    """
    return cast(RefundCase, case)


def build_refund_gates(config: dict | None = None) -> list[HardGate]:
    """Build the ordered refund hard-gate list.

    Args:
        config: Optional thresholds dict. Recognised keys (all optional)::

            {
              "RG_FRAUD":         {"risk_threshold": 0.7},
              "RG_NO_EVIDENCE":   {"completeness_threshold": 0.15},
              "RG_LARGE_AMOUNT":  {"amount_threshold_usd": 2000},
              "RG_ACCOUNT_SWITCH":{"risk_threshold": 0.5},
            }

    Returns:
        Ordered list of :class:`HardGate`. Evaluated in order; first match wins.
    """
    cfg = config or {}
    fraud_risk = cfg.get("RG_FRAUD", {}).get("risk_threshold", DEFAULT_FRAUD_RISK_THRESHOLD)
    no_evidence = cfg.get("RG_NO_EVIDENCE", {}).get(
        "completeness_threshold", DEFAULT_NO_EVIDENCE_THRESHOLD
    )
    amount_limit = cfg.get("RG_LARGE_AMOUNT", {}).get(
        "amount_threshold_usd", DEFAULT_AMOUNT_LIMIT_USD
    )
    switch_risk = cfg.get("RG_ACCOUNT_SWITCH", {}).get(
        "risk_threshold", DEFAULT_ACCOUNT_SWITCH_RISK_THRESHOLD
    )

    return [
        HardGate(
            gate_id="RG_POLICY",
            description="Outside refund window or non-refundable item -> DECLINE",
            condition=lambda c: (not _rc(c).within_policy_window or not _rc(c).item_returnable),
            forced_decision=Decision.DECLINE,
            rationale_template=(
                "Refund gate RG_POLICY triggered: the request falls outside the "
                "refund policy window or concerns a non-refundable item, so it "
                "is declined as a pure policy rule. This is independent of the "
                "${amount:,.2f} amount or the abuse-risk signal ({risk_score:.3f}); "
                "the request would only qualify if it were submitted within the "
                "policy window for a refundable item."
            ),
        ),
        HardGate(
            gate_id="RG_FRAUD",
            description="Fraud suspected with high abuse risk -> DECLINE",
            condition=lambda c: (_rc(c).fraud_suspected and _rc(c).abuse_risk > fraud_risk),
            forced_decision=Decision.DECLINE,
            rationale_template=(
                "Refund gate RG_FRAUD triggered: a fraud signal fired and the "
                "abuse-risk score ({risk_score:.3f}) exceeds the refund fraud "
                "threshold of "
                + f"{fraud_risk:.2f}"
                + ", so the ${amount:,.2f} refund is declined automatically. A "
                "favourable resolution would require the fraud signal to be "
                "cleared and the abuse risk reduced below the threshold."
            ),
        ),
        HardGate(
            gate_id="RG_NO_EVIDENCE",
            description="Evidence essentially absent -> DEFER (ask for proof)",
            condition=lambda c: _rc(c).evidence_completeness < no_evidence,
            forced_decision=Decision.DEFER,
            rationale_template=(
                "Refund gate RG_NO_EVIDENCE triggered: evidence completeness "
                "({completeness:.3f}) is below the minimum of "
                + f"{no_evidence:.2f}"
                + ", so instead of guessing on the ${amount:,.2f} request the "
                "system defers and asks the customer for supporting evidence "
                "(receipt, order number, photo). The request can be reconsidered "
                "once the missing documentation is provided."
            ),
        ),
        HardGate(
            gate_id="RG_LARGE_AMOUNT",
            description="Refund above auto-approval limit -> ESCALATE",
            condition=lambda c: _rc(c).refund_amount > amount_limit,
            forced_decision=Decision.ESCALATE,
            rationale_template=(
                "Refund gate RG_LARGE_AMOUNT triggered: the requested refund "
                "(${amount:,.2f}) exceeds the auto-approval limit of $"
                + f"{amount_limit:,.0f}"
                + ", so the case is routed to a human reviewer (ESCALATE) rather "
                "than being released automatically. A reviewer may approve the "
                "full amount after confirming the order and the evidence."
            ),
        ),
        HardGate(
            gate_id="RG_DISPUTE",
            description="Prior dispute/chargeback on relationship -> ESCALATE",
            condition=lambda c: _rc(c).chargeback_prior,
            forced_decision=Decision.ESCALATE,
            rationale_template=(
                "Refund gate RG_DISPUTE triggered: a prior dispute/chargeback "
                "exists on this relationship, so the ${amount:,.2f} request is "
                "escalated for human review regardless of the abuse-risk score "
                "({risk_score:.3f}). Prior dispute history requires a reviewer to "
                "assess the pattern before any refund is released."
            ),
        ),
        HardGate(
            gate_id="RG_ACCOUNT_SWITCH",
            description="Destination account changed with elevated risk -> ESCALATE",
            condition=lambda c: (
                _rc(c).destination_account_changed and _rc(c).abuse_risk > switch_risk
            ),
            forced_decision=Decision.ESCALATE,
            rationale_template=(
                "Refund gate RG_ACCOUNT_SWITCH triggered: the refund destination "
                "account differs from the source and the abuse-risk score "
                "({risk_score:.3f}) exceeds "
                + f"{switch_risk:.2f}"
                + ", a common account-takeover pattern, so the ${amount:,.2f} "
                "request is escalated for human verification of the destination "
                "account before any funds move."
            ),
        ),
    ]
