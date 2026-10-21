# Artifact: Placement-Aware Evaluation (anonymous submission)

Code and aggregated result artifacts for the submitted paper. Provided for double-blind review.

## Contents

- `scripts/` — experiment and figure scripts (Python 3.10).
  - `r2_e11_lanl_blind.py` — LANL training/scoring used to build the evaluation artifacts (requires GPU; results provided, no need to rerun).
  - `r3_e25_attack_bed.py` + `r3_e31_lib.py` — attacker bed: declared placement family, best responses, frozen criteria.
  - `r3_e32_cas_stratified.py`, `r3_e32b_flip_ci.py` — placement sweeps, three metric views, CAS criteria.
  - `r3_e33_a4prime_adaptive.py` — information-symmetric audit (A4′) vs mismatched A4.
  - `r4_e35_lanl_placement.py` — LANL real-event placement reweighting (CPU, exploratory).
  - `make_figs.py`, `make_fig1.py` — regenerate all paper figures from `results/`.
- `results/` — aggregated artifacts (JSON/JSONL) from which every number in the paper is derived. Raw per-event score files (`.npz`) are omitted for size; the aggregations below are the paper's source of record.

## Reproducing

All headline numbers regenerate in seconds on CPU from `results/`:

```
python scripts/r4_e35_lanl_placement.py        # Table II last row, Sec. VII reweighting audit
python scripts/make_figs.py                    # Figs. 2-4 (reads results/r3/...; adjust ROOT)
python scripts/make_fig1.py                    # Fig. 1 schematic
```

`r3_*` scripts read the E25 bed artifacts; `stratified_table.jsonl` and `results_e33.jsonl` in `results/` are their recorded outputs, so every pooled/standardized/F value, per-seed value, CI, and occupancy census in the paper can be recomputed from these files directly (pandas one-liners; see script `metrics_all`/`triple` in `r3_e32_cas_stratified.py` for the exact definitions).

Scripts assume the project root is `/root/autodl-tmp` (see `sys.path` lines); adjust `ROOT`/path constants to your local layout. Training-scale experiments (E11/E25) require a GPU and the LANL authentication dataset; all confirmatory numbers reported in the paper are included here as artifacts and need no rerun.

## Provenance notes

- Frozen criteria files (`criteria_frozen.json`) were written before execution per the project's preregistration discipline; E35 (`STATUS_addendum.json`) is exploratory with no frozen criteria and is labeled as such in the paper.
- E35 reads recorded scores from the LANL blind-arm artifacts; it performs no training.
- The standardized view is the unweighted mean of within-stratum AUCs over occupied strata (degree quintiles, K=5, per-day); this convention matches the analysis code exactly.
