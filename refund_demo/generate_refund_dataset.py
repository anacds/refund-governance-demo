# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Generate the synthetic refund dataset to JSONL.

Builds ~50 deterministic + boundary + core ``RefundCase`` records with a fixed
seed and writes them to ``data/refund_cases.jsonl``.

Ground truth is tied directly to the refund policy:
  * If a refund hard gate fires, the case is DETERMINISTIC and its ``gt_decision``
    is the gate's forced decision (the policy has a single defensible answer).
  * Otherwise the case is AMBIGUOUS and gets a ``gt_decision_set`` of defensible
    resolutions (used for deferral/over-caution metrics, not accuracy).

Synthetic data only — this is a governance/evaluation demo, not a refund engine.

Example::

    python refund_demo/generate_refund_dataset.py --seed 42 \
        --out refund_demo/data/refund_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Allow running directly as a script (python refund_demo/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mech_gov.data.banking_case import Decision, GTConfidence, TransactionType  # noqa: E402
from mech_gov.governance.primitives.hard_gates import evaluate_hard_gates  # noqa: E402
from refund_demo.refund_case import RefundCase  # noqa: E402
from refund_demo.refund_gates import build_refund_gates  # noqa: E402

_DEFAULT_OUT = str(Path(__file__).resolve().parent / "data" / "refund_cases.jsonl")


def _case(case_id: str, **overrides: Any) -> RefundCase:
    """Build a RefundCase with refund-friendly defaults."""
    base: dict[str, Any] = dict(
        case_id=case_id,
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.30,
        completeness=0.70,
        amount_usd=300.0,
        within_policy_window=True,
        item_returnable=True,
    )
    base.update(overrides)
    return RefundCase(**base)


def _deterministic_cases() -> list[RefundCase]:
    """One or more cases that each cleanly trigger a single refund gate."""
    return [
        # RG_POLICY -> DECLINE
        _case("refund-policy-window-01", within_policy_window=False, amount_usd=120.0),
        _case("refund-policy-window-02", within_policy_window=False, amount_usd=540.0),
        _case("refund-policy-return-01", item_returnable=False, amount_usd=80.0),
        # RG_FRAUD -> DECLINE (fraud signal + high abuse risk)
        _case("refund-fraud-01", fraud_suspected=True, risk_score=0.82, amount_usd=450.0),
        _case("refund-fraud-02", fraud_suspected=True, risk_score=0.91, amount_usd=130.0),
        # RG_NO_EVIDENCE -> DEFER (evidence essentially absent)
        _case("refund-noevidence-01", completeness=0.05, amount_usd=220.0),
        _case("refund-noevidence-02", completeness=0.10, amount_usd=75.0),
        # RG_LARGE_AMOUNT -> ESCALATE (above auto-approval limit)
        _case("refund-large-01", amount_usd=3500.0, risk_score=0.30),
        _case("refund-large-02", amount_usd=2500.0, risk_score=0.20),
        # RG_DISPUTE -> ESCALATE (prior chargeback)
        _case("refund-dispute-01", chargeback_prior=True, amount_usd=400.0),
        _case("refund-dispute-02", chargeback_prior=True, amount_usd=900.0, risk_score=0.25),
        # RG_ACCOUNT_SWITCH -> ESCALATE (destination changed + elevated risk)
        _case(
            "refund-acctswitch-01",
            destination_account_changed=True,
            risk_score=0.66,
            amount_usd=300.0,
        ),
        _case(
            "refund-acctswitch-02",
            destination_account_changed=True,
            risk_score=0.58,
            amount_usd=150.0,
        ),
    ]


def _boundary_cases() -> list[RefundCase]:
    """Cases just on the safe side of a threshold (no gate fires)."""
    return [
        # Amount just below the $2,000 auto-approval limit.
        _case("refund-boundary-amount-01", amount_usd=1_990.0, completeness=0.72),
        _case("refund-boundary-amount-02", amount_usd=1_950.0, completeness=0.60, risk_score=0.45),
        # Evidence just above the no-evidence floor (0.15) but still thin:
        # below the ambiguity threshold (0.30), so R2's post-LLM gate should bite.
        _case("refund-boundary-evidence-01", completeness=0.18, amount_usd=260.0, risk_score=0.35),
        _case("refund-boundary-evidence-02", completeness=0.22, amount_usd=180.0, risk_score=0.20),
        # Fraud signal but abuse risk just below the 0.70 fraud threshold.
        _case("refund-boundary-fraud-01", fraud_suspected=True, risk_score=0.68, amount_usd=300.0),
        # Account changed but risk just below the 0.50 switch threshold.
        _case(
            "refund-boundary-switch-01",
            destination_account_changed=True,
            risk_score=0.48,
            amount_usd=250.0,
        ),
    ]


def _core_cases(rng: random.Random, n: int) -> list[RefundCase]:
    """Randomised 'core' cases with no gate firing — the LLM decides."""
    cases: list[RefundCase] = []
    for i in range(n):
        amount = round(rng.uniform(40.0, 1_800.0), 2)
        completeness = round(rng.uniform(0.35, 0.95), 3)
        risk = round(rng.uniform(0.05, 0.65), 3)
        prior = rng.choice([0, 0, 1, 2])
        cases.append(
            _case(
                f"refund-core-{i:02d}",
                amount_usd=amount,
                completeness=completeness,
                risk_score=risk,
                prior_refunds_30d=prior,
            )
        )
    return cases


def _refund_decision_set(case: RefundCase) -> list[str]:
    """Defensible resolutions for an ambiguous (non-gated) refund request."""
    acceptable: set[str] = set()

    # Low risk + good evidence + modest amount -> approving is defensible.
    if case.abuse_risk < 0.4 and case.evidence_completeness > 0.6 and case.refund_amount < 1_000:
        acceptable.add(Decision.APPROVE.value)
    # Decent evidence -> a conditional resolution is defensible.
    if case.evidence_completeness > 0.4:
        acceptable.add(Decision.CONDITIONAL.value)
    # Elevated risk or many recent refunds -> escalation is defensible.
    if case.abuse_risk > 0.45 or case.prior_refunds_30d >= 2:
        acceptable.add(Decision.ESCALATE.value)
    # Thin evidence -> deferring to ask for proof is defensible.
    if case.evidence_completeness < 0.5:
        acceptable.add(Decision.DEFER.value)
    # DEFER is always a minimally defensible conservative choice.
    acceptable.add(Decision.DEFER.value)
    return sorted(acceptable)


def assign_refund_ground_truth(
    cases: list[RefundCase], gates_config: dict | None = None
) -> list[RefundCase]:
    """Assign ground truth using the refund policy itself.

    Gate-firing cases are deterministic (single defensible decision = the gate's
    forced decision); all other cases are ambiguous with a defensible set.
    """
    gates = build_refund_gates(gates_config)
    for case in cases:
        gate_result = evaluate_hard_gates(case, gates, gates_config)
        if gate_result is not None:
            gate_id, forced, _rationale = gate_result
            case.gt_decision = forced.value
            case.gt_decision_set = [forced.value]
            case.gt_confidence = GTConfidence.DETERMINISTIC
            case.gt_rule_id = gate_id
            case.gt_rationale = f"Refund gate {gate_id} yields a single defensible decision."
        else:
            case.gt_decision = None
            case.gt_decision_set = _refund_decision_set(case)
            case.gt_confidence = GTConfidence.AMBIGUOUS
            case.gt_rule_id = None
            case.gt_rationale = "No refund gate fires; multiple resolutions are defensible."
    return cases


def build_refund_cases(seed: int = 42, n_core: int = 24) -> list[RefundCase]:
    """Build the full, deterministic refund dataset (with ground truth)."""
    rng = random.Random(seed)
    cases = _deterministic_cases() + _boundary_cases() + _core_cases(rng, n_core)
    for case in cases:
        case.seed = seed
    assign_refund_ground_truth(cases)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic refund dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-core", type=int, default=24, help="number of randomised core cases")
    parser.add_argument("--out", default=_DEFAULT_OUT)
    args = parser.parse_args()

    cases = build_refund_cases(seed=args.seed, n_core=args.n_core)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")

    n_det = sum(1 for c in cases if c.gt_confidence == GTConfidence.DETERMINISTIC)
    print(f"Wrote {len(cases)} refund cases ({n_det} deterministic) to {out_path}")


if __name__ == "__main__":
    main()
