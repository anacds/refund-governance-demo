# Refund governance demo

A worked example of **mechanically governing a refund (chargeback) decision**
with an LLM. An LLM reads a refund request and recommends a resolution, but
deterministic code around the model decides the sensitive points (large amount,
suspected fraud, missing evidence, out-of-policy). It runs against an **OpenAI**
model through a vendor-neutral, standard-library HTTP client.

> This is a **governance / evaluation demo on synthetic data** — not a
> production refund engine. No real customer data is involved.

## Built on `mech-gov-framework`

This project is a **use-case application** of the open-source
[`mech-gov-framework`](https://github.com/SantanderAI/mech-gov-framework)
("Mechanical Governance for LLM Decisions") by Santander AI Lab, licensed under
Apache-2.0. Because the framework is not published on PyPI, its source is
vendored here under `src/mech_gov/` (unmodified). All attribution and license
terms are preserved — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

> **Vendored provenance:** `src/mech_gov/` was copied from
> [`mech-gov-framework`](https://github.com/SantanderAI/mech-gov-framework) at
> commit `bfcff58`. Only the files under `refund_demo/` are this
> project's own work.

The demo reuses the framework's governance regimes and primitives **without
reimplementing them**:

- **R1 (text-only):** the policy is in the prompt; the model is trusted.
- **R2 (mechanical):** policy + hard gates → E3 commit → CEFL → I6Q →
  ambiguity gate → E3 reveal.
- **R3 (adaptive):** R2 + bounded, safe self-modification (invariants + drift
  budget). *Exploratory in the framework.*

## Quickstart

The demo calls **OpenAI**. Set your API key first:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # installs the vendored framework
# pip install -e ".[dev]"          # optional: ruff / black / mypy

export OPENAI_API_KEY=sk-...        # required
# export OPENAI_MODEL=gpt-4o-mini   # optional (this is the default)

# generate the synthetic dataset and run the refund demo
python refund_demo/generate_refund_dataset.py
python refund_demo/run_refund_demo.py
# cap API cost while iterating:
python refund_demo/run_refund_demo.py --limit 8
# see what happens per case (decision + gates), and the framework's own logs:
python refund_demo/run_refund_demo.py --limit 8 --verbose --debug
```

The runner narrates each step (`▶ [k/6] ...`) with a live `case k/N` counter, so
it is clear what is running. `--verbose` prints one line per case; `--debug`
unmutes the framework's per-primitive logs. `OPENAI_API_KEY` can also be placed
in a repo-root `.env` (already git-ignored) instead of exporting it.

The demo prints an R1-vs-R2 metrics table, a persuasive-framing sub-experiment
(FSR), an R3 self-modification sub-experiment, and a hard-gate invariant check.

## Example results

A full run on `gpt-4o-mini` over the 43-case synthetic dataset. Numbers vary
from run to run (CEFL samples at temperature > 0); the hard-gate invariant holds
on every run.

| Metric (13 deterministic cases) | R1 (text-only) | R2 (mechanical) |
| --- | --- | --- |
| accuracy | 0.54 | **1.00** |
| macro-F1 | 0.50 | **1.00** |
| MCC | 0.45 | **1.00** |
| framing flip rate (FSR, lower = better) | 0.26 | **0.14** |

- **Hard rules beat prompting.** On cases with a single defensible answer, R2 is
  correct every time because hard gates decide them *before* the LLM. R1 — which
  only has the policy in its prompt — gets nearly half wrong.
- **Framing robustness.** Rewriting each request in an emotional, persuasive tone
  (same facts, different words) flips ~26% of R1's decisions but only ~14% of
  R2's. R2 cannot fully neutralise framing on cases that still reach the LLM, but
  it *guarantees* stability on the gated (sensitive) cases.
- **The invariant held.** Across all 43 cases, R2 never approved a request that a
  prohibitive gate (out-of-policy / suspected fraud) had already blocked.
- **R3 in this run.** The model proposed no policy modifications, so the drift
  budget was untouched and the refund invariants were trivially preserved. The
  R3 machinery runs; whether a model proposes changes depends on the model.

## What's in here

| Path | What it is |
| --- | --- |
| `refund_demo/` | **the project** — refund use case (models, gates, regimes, dataset, runner) |
| `src/mech_gov/` | vendored upstream library (Santander, Apache-2.0), unmodified |
| `LICENSE`, `NOTICE` | Apache-2.0 license and attribution for the framework |

See [`refund_demo/README.md`](refund_demo/README.md) for the
full walkthrough of the refund demo, the gates, the metrics, and what to look at.

## Backend notes

The demo reaches OpenAI through the framework's standard-library
`openai_compatible` provider — **no OpenAI/cloud SDK is imported**. You can point
it at any OpenAI-compatible endpoint (Azure OpenAI, a gateway, a local server)
by overriding `OPENAI_BASE_URL`. The framework also ships an offline `mock`
provider for deterministic, no-credential runs.

## License & attribution

This repository builds on `mech-gov-framework` by Santander AI Lab, used under
the Apache License 2.0. The vendored framework source under `src/mech_gov/`
retains its original license headers and the project-level `LICENSE` and
`NOTICE` are kept intact, as required by the license.
