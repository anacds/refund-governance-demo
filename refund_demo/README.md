<!--
Copyright (c) 2026 Santander Group
SPDX-License-Identifier: Apache-2.0
-->

# Refund approval demo — mechanical governance of a single decision

This example shows **mechanical governance of a refund (chargeback) decision**.
An LLM reads a refund request and recommends a resolution, but deterministic
code around the model decides the sensitive points (large amount, suspected
fraud, missing evidence, out-of-policy). It runs against **OpenAI**.

> This is a **governance / evaluation demo on synthetic data** — not a
> production refund engine. No real customer data is involved.

It contrasts the framework's regimes:

- **R1 (text-only)** — the refund policy is written into the system prompt and
  the model is trusted to follow it. No mechanical enforcement.
- **R2 (mechanical)** — the same policy, plus the framework's mechanical
  pipeline: pre-LLM **hard gates** → E3 commit → **CEFL** (candidate freezing)
  → **I6Q** (argument-quality) → **ambiguity gate** → E3 reveal. Only the
  refund gates and the policy prompt change; every primitive is reused unchanged.
- **R3 (adaptive)** — R2 plus *bounded, safe self-modification*: the model may
  propose refund-policy changes, accepted into a vetted backlog only if they
  break no **invariant** and fit a **drift budget** (see below).

## How to run

The demo calls **OpenAI** (via the framework's stdlib `openai_compatible`
provider — no OpenAI SDK is imported).

```bash
# from the repository root, with the package installed (pip install -e ".[dev]")
export OPENAI_API_KEY=sk-...                                # required (or put it in a repo-root .env)
# export OPENAI_MODEL=gpt-4o-mini                           # optional (default)

python refund_demo/generate_refund_dataset.py      # writes data/refund_cases.jsonl
python refund_demo/run_refund_demo.py              # runs the full demo
python refund_demo/run_refund_demo.py --limit 8    # cap API cost while iterating
python refund_demo/run_refund_demo.py --verbose    # one line per case (decision + gates)
python refund_demo/run_refund_demo.py --debug      # also show the framework's own logs
```

The runner narrates each step (`▶ [k/6] ...`) with a live `case k/N` counter and
per-phase timing, so it is clear what is running and roughly how long it takes.

The demo prints four blocks: an R1-vs-R2 metrics table, a persuasive-framing
sub-experiment (FSR), an R3 self-modification sub-experiment, and a hard-gate
invariant check.

You can point at any OpenAI-compatible endpoint (Azure OpenAI, a gateway, a
local server) by overriding `OPENAI_BASE_URL`.

## What to look at

1. **The hard-gate invariant.** R2 **never** emits `APPROVE` on a case where a
   prohibitive gate (`RG_POLICY`, `RG_FRAUD`) fired. This holds regardless of
   the model — the gates decide before the LLM is ever called.
2. **Task metrics** (accuracy / macro-F1 / MCC on deterministic cases): R2
   matches the policy because the gates decide those cases mechanically, while
   R1 depends entirely on the model's judgement.
3. **Framing robustness (FSR):** the same request written in a neutral vs an
   emotional/persuasive tone — same numbers. A customer asking for a refund
   writes emotionally all the time; the decision should not change because of
   that. R2 keeps gated cases stable regardless of wording, so its FSR should be
   at or below R1's.

> Note: results from a real model are non-deterministic (CEFL samples at
> temperature > 0), so exact numbers vary run to run. The hard-gate invariant,
> however, holds on every run.

## R3: bounded, safe self-modification (adaptive)

R3 is **exploratory** in the framework (a stub kept as a forward reference), so
treat this section as a preview rather than the core result.

On top of the full R2 pipeline, the model may *propose* a refund-policy change
(e.g. "raise the auto-approval limit to $2,500 for low-risk, well-evidenced
requests"). The framework decides whether to accept each proposal into a vetted
backlog using two guards:

1. **Refund invariants** — sacred rules no modification may break:
   - `INV_REFUND_1`: suspected fraud + high abuse risk → never `APPROVE`
   - `INV_REFUND_2`: out of policy window / non-refundable → never `APPROVE`
   - `INV_REFUND_3`: essentially no evidence → never `APPROVE`/`CONDITIONAL`
2. **Drift budget** — a cumulative cap (default `1.0`); each accepted proposal
   consumes its `cost`. Once exhausted, further proposals are rejected. This
   stops the policy from slowly drifting toward "approve everything".

Important: R3 here **harvests bounded policy-improvement proposals** — the case
decision itself stays the R2 decision. The intended production shape is a
human-in-the-loop policy copilot: **R2 decides live; R3 runs over history and
emits a safety-vetted backlog of policy tweaks for a human to review.**

The demo's R3 block reports how many proposals the model made, how many were
accepted (within the drift budget and breaking no invariant) vs rejected, with
the refund invariants preserved on every case.

## Refund hard gates (first match wins, evaluated before the LLM)

| Gate | Condition | Decision |
| --- | --- | --- |
| `RG_POLICY` | outside refund window **or** non-refundable item | `DECLINE` |
| `RG_FRAUD` | fraud suspected **and** abuse risk > 0.70 | `DECLINE` |
| `RG_NO_EVIDENCE` | evidence completeness < 0.15 | `DEFER` |
| `RG_LARGE_AMOUNT` | refund amount > $2,000 | `ESCALATE` |
| `RG_DISPUTE` | prior dispute/chargeback on the relationship | `ESCALATE` |
| `RG_ACCOUNT_SWITCH` | destination account changed **and** abuse risk > 0.50 | `ESCALATE` |

All thresholds come from config (`build_refund_gates(config)`) with the defaults
above. When no gate fires, the LLM decides — and CEFL, I6Q and the ambiguity
gate still constrain the outcome (e.g. an I6Q failure forces `ESCALATE`; thin
evidence below the ambiguity threshold forces `DEFER`/`ESCALATE`).

## Data model

`RefundCase` subclasses `BankingCase` and reuses its inherited numeric fields so
the existing primitives and metrics work unchanged:

| Inherited field | Refund meaning | Alias |
| --- | --- | --- |
| `amount_usd` | requested refund amount | `refund_amount` |
| `risk_score` | abuse / fraud risk (0..1) | `abuse_risk` |
| `completeness` | evidence completeness (0..1) | `evidence_completeness` |

Refund-specific policy lives in new fields (`fraud_suspected`,
`chargeback_prior`, `within_policy_window`, `item_returnable`,
`prior_refunds_30d`, `destination_account_changed`) — not in
`regulatory_flags`, whose validator only accepts the fixed banking flag universe.

## Decision meanings (refund context)

- `APPROVE` — refund the full amount
- `CONDITIONAL` — partial refund, store credit, or refund on return of the item
- `ESCALATE` — route to a human reviewer
- `DEFER` — ask for more information (receipt, order number, photo)
- `DECLINE` — refuse the refund

## Files

| File | What it is |
| --- | --- |
| `refund_case.py` | `RefundCase(BankingCase)` + neutral/persuasive `to_prompt()` |
| `refund_gates.py` | `build_refund_gates(config)` |
| `refund_regimes.py` | `RefundR2`, `RefundR3` + R1 wiring + refund invariants + policy loader |
| `refund_policy.txt` | refund policy prompt (same format as `policy_templates/`) |
| `generate_refund_dataset.py` | builds the seeded synthetic dataset + ground truth |
| `run_refund_demo.py` | R1-vs-R2 comparison + framing + invariant check |
| `data/refund_cases.jsonl` | generated dataset (fixed seed) |

Determinism: the dataset build and the `entropy_seed` are seeded, so the dataset
and the hard-gate decisions are reproducible. The LLM-driven parts use a real
model (temperature > 0 inside CEFL), so their exact numbers vary run to run.
