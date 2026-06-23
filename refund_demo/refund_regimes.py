# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Refund governance regimes (R1 text-only, R2 mechanical, R3 adaptive).

All three regimes are thin wrappers over the framework's existing regimes; the
only changes are the system prompt (the refund policy) and, for R2/R3, the
hard-gate list (and, for R3, the safety invariants). Every mechanical primitive
(E3 commit-reveal, CEFL, I6Q, ambiguity gate, reveal) and the R3 self-
modification machinery (invariant checks + drift budget) are inherited unchanged.

The refund policy is loaded the same way the bundled policy templates are: by
reading a ``.txt`` file (here from this example directory rather than the
packaged ``policy_templates/`` folder).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.governance import r3_adaptive as _r3
from mech_gov.governance.primitives.i6q import I6QConfig
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.r3_adaptive import Invariant, R3Adaptive
from refund_demo.refund_case import RefundCase
from refund_demo.refund_gates import (
    DEFAULT_FRAUD_RISK_THRESHOLD,
    DEFAULT_NO_EVIDENCE_THRESHOLD,
    build_refund_gates,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent


def load_refund_template(name: str = "refund_policy") -> str:
    """Load a refund policy template by filename (without extension).

    Mirrors :func:`mech_gov.governance.policy_templates.load_template`, but reads
    from this example directory so the demo stays self-contained.
    """
    path = _TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Refund template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_refund_r1(temperature: float = 0.7) -> R1TextOnly:
    """R1 (refund): reuse text-only governance with the refund policy prompt.

    Represents "I wrote the rule in the prompt and trust the model" — no
    mechanical enforcement at all.
    """
    regime = R1TextOnly(temperature=temperature)
    regime._system_prompt = load_refund_template()  # noqa: SLF001 (intentional override)
    return regime


class RefundR2(R2Mechanical):
    """R2 (refund): mechanical governance with refund gates and the refund policy.

    Inherits the full R2 pipeline (hard gates -> E3 commit -> CEFL -> I6Q ->
    ambiguity gate -> E3 reveal). The constructor swaps in the refund hard gates
    and the refund policy prompt after the base initialisation.
    """

    def __init__(
        self,
        refund_gates_config: dict | None = None,
        i6q_config: I6QConfig | None = None,
        n_cefl_candidates: int = 3,
        theta_iota: float = 0.3,
        risk_escalation_threshold: float = 0.7,
    ):
        super().__init__(
            i6q_config=i6q_config,
            n_cefl_candidates=n_cefl_candidates,
            theta_iota=theta_iota,
            risk_escalation_threshold=risk_escalation_threshold,
        )
        # Point the prompt at the refund policy and replace the banking gates
        # with the refund gates (everything else stays inherited).
        self._system_prompt = load_refund_template()
        self._gates = build_refund_gates(refund_gates_config)
        self._gates_config = refund_gates_config


# ---------------------------------------------------------------------------
# R3: refund safety invariants
# ---------------------------------------------------------------------------
#
# Invariants are the "sacred" rules that NO policy modification may ever break.
# They are a backstop: under R2's refund gates the decision already respects
# them, so in practice the invariant checks should always pass. The framework
# looks up each check by name in ``r3_adaptive.INVARIANT_CHECKS``, so we register
# our functions there at import time.


def check_inv_refund_fraud(case: BankingCase, decision: Decision) -> str | None:
    """INV_REFUND_1: suspected fraud with high abuse risk -> never APPROVE."""
    rc = cast(RefundCase, case)
    if (
        rc.fraud_suspected
        and rc.abuse_risk > DEFAULT_FRAUD_RISK_THRESHOLD
        and decision == Decision.APPROVE
    ):
        return "INV_REFUND_1 violated: APPROVE with suspected fraud and high abuse risk"
    return None


def check_inv_refund_policy(case: BankingCase, decision: Decision) -> str | None:
    """INV_REFUND_2: out of policy window or non-refundable item -> never APPROVE."""
    rc = cast(RefundCase, case)
    if (not rc.within_policy_window or not rc.item_returnable) and decision == Decision.APPROVE:
        return "INV_REFUND_2 violated: APPROVE outside policy window / non-refundable item"
    return None


def check_inv_refund_no_evidence(case: BankingCase, decision: Decision) -> str | None:
    """INV_REFUND_3: essentially no evidence -> must DEFER (never APPROVE/CONDITIONAL)."""
    rc = cast(RefundCase, case)
    if rc.evidence_completeness < DEFAULT_NO_EVIDENCE_THRESHOLD and decision in {
        Decision.APPROVE,
        Decision.CONDITIONAL,
    }:
        return f"INV_REFUND_3 violated: {decision.value} with essentially no evidence"
    return None


# Register the checks so the framework's name-based lookup can find them.
_r3.INVARIANT_CHECKS["check_inv_refund_fraud"] = check_inv_refund_fraud
_r3.INVARIANT_CHECKS["check_inv_refund_policy"] = check_inv_refund_policy
_r3.INVARIANT_CHECKS["check_inv_refund_no_evidence"] = check_inv_refund_no_evidence

REFUND_INVARIANTS: list[Invariant] = [
    Invariant(
        invariant_id="INV_REFUND_1",
        description="Suspected fraud + high abuse risk -> never APPROVE",
        check_fn="check_inv_refund_fraud",
    ),
    Invariant(
        invariant_id="INV_REFUND_2",
        description="Out of policy window / non-refundable item -> never APPROVE",
        check_fn="check_inv_refund_policy",
    ),
    Invariant(
        invariant_id="INV_REFUND_3",
        description="No evidence -> never APPROVE/CONDITIONAL (must DEFER)",
        check_fn="check_inv_refund_no_evidence",
    ),
]


class RefundR3(R3Adaptive):
    """R3 (refund): R2 plus bounded, safe self-modification.

    On top of the full R2 refund pipeline, the model may *propose* refund-policy
    modifications (e.g. "raise the auto-approval limit for low-risk loyal
    customers"). Each proposal is accepted into a vetted backlog only if:

      * it does not break any refund invariant (:data:`REFUND_INVARIANTS`), and
      * it fits within the remaining drift budget.

    The case decision itself is still the R2 decision; R3 harvests bounded
    policy-improvement proposals rather than rewriting behaviour mid-stream.
    """

    def __init__(
        self,
        refund_gates_config: dict | None = None,
        drift_budget_max: float = 1.0,
        i6q_config: I6QConfig | None = None,
        n_cefl_candidates: int = 3,
        theta_iota: float = 0.3,
        risk_escalation_threshold: float = 0.7,
    ):
        super().__init__(
            drift_budget_max=drift_budget_max,
            invariants=REFUND_INVARIANTS,
            i6q_config=i6q_config,
            n_cefl_candidates=n_cefl_candidates,
            theta_iota=theta_iota,
            risk_escalation_threshold=risk_escalation_threshold,
        )
        # Refund policy prompt + the R3 suffix that invites a modification proposal.
        self._system_prompt = load_refund_template() + _r3.MODIFICATION_PROMPT_SUFFIX
        self._gates = build_refund_gates(refund_gates_config)
        self._gates_config = refund_gates_config
