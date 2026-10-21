# Accelerating Batch Prediction Intervals for CP-MDA-Nested

Code and reproducibility materials for Peng Huang's CDICS 2026 paper. The method reuses sorted calibration-score groups, computes query-dependent group offsets, selects the adjacent target ranks implicitly, and applies the reference linear quantile rule.

## Quick start

Use Python 3.12 (the archived environment is Python 3.12.14). From a fresh clone:

```sh
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
```

First execution compiles Numba kernels and may take substantially longer than a warm query. The test performs 106 fixtures (5,424 endpoint comparisons), 483 independent rank checks, and all five computation variants on each station's 5,832 frozen query rows. Expected outcome: maximum fixture difference 0 and ten real-data exact-match entries. It writes correctness.json and verification_local.json. No model fitting or query-response access occurs.

Only load the two trusted model files from this repository after checking SHA256SUMS.json. joblib uses pickle and can execute code when loading arbitrary untrusted files. Verification checks model/input SHA256 before loading.

## Main files

- selector.py: multiplicity-aware rank selection on shifted sorted arrays and linear interpolation.
- experiment.py: explicit NumPy/dense/chunked controls, synthetic fixtures and correctness checks. Its optional process-memory helper is Windows-specific; verification does not use it.
- benchmark.py: joint-mask cache construction and mixed-query batch evaluation.
- pilot.py: the original preprocessing compatibility class for the frozen model files.
- verify.py: independent numerical reproduction entry point; verification_reference.json records the successful public-export verification.
- Aotizhongxin/Dongsi .npz and .joblib: frozen inputs and minimal exported models; model_export_checks.json documents the export.
- repeat_timing.json: all 20 randomized five-method blocks at each station.
- supplemental_checks.json and scale_all_54.csv: all 54 n/B/G/alpha configurations and the artificial two-sensor-outage test.
- cold_process_summary.json and amortization_estimate.json: recorded separate-process cold costs and same-protocol amortization estimates.
- cache_workload.json, author_commit.json and discard_invariant_checks.json: workload, pinned statistical-reference version and instrumented implementation checks.

## Optional timing reruns

```sh
python run_benchmark.py
python run_supplemental.py
```

These scripts can take several minutes; they write new observations under runs/ and refuse to overwrite an existing run. run_benchmark.py uses the original Windows affinity/memory interface and requires Windows. The supplemental runner does not require that interface. Archived timings remain at the repository root. Random seeds and protocols are in benchmark_protocol.md and supplemental_protocol.md. Wall-clock times and the identity of the fastest baseline may vary on another machine. The public model export preserves predictions but may alter file-loading and unpickling costs: the supplied historical cold measurements refer to the original untrimmed artifacts and are not promised for the public exports.

## Recorded findings and interpretation

The revised pinned-core benchmark reports warm reductions of 21.65%/34.71% against the fastest tested presorted C256 control, with descriptive paired-block bootstrap intervals. Endpoint agreement is computational; it does not establish coverage on dependent time-series data. Initial preparation/JIT and memory costs remain. The 54-case grid contains both gains and losses; no uniform speedup is claimed. Selection itself is established algorithmic machinery; the contribution is CP-MDA-specific representation and reuse.

## Licensing and provenance

Author-written code and model artifacts: MIT. Derived UCI inputs: CC BY 4.0, with attribution and transformations in DATA_LICENSE.md. Upstream statistical-method attribution and the original MIT notice are in THIRD_PARTY_NOTICES.md and UPSTREAM_LICENSE_MIT.txt. There are no editor emails, reviewer forms, manuscript PDFs, local account details or unrelated project files in this repository. Local absolute paths in archived JSON records were sanitized; mathematical and timing values were retained.

This release corresponds to the revised paper. GitHub's commit identifier pins the actual code; check CITATION.cff for software citation information. The source manuscript is supplied separately to the conference.
