# Pyrova

A differentiable macro placer for thermal tail risk under uncertain workload power.

## Setup

    python -m venv .venv && source .venv/bin/activate
    make install

## Verify

    make verify

Two gates, both must pass before any change to the numerics lands:

- `pyrova/tests/golden.py` — bit-level snapshot of the solver field, peak,
  adjoint gradient, and placer objectives. Per-platform (BLAS-dependent);
  regenerate on a new machine with `make golden`.
- `pyrova/tests/test_gradients.py` — finite-difference checks of every
  gradient path (max-error asserts, subgradient kinks detected and exempted).

## Layout

    pyrova/thermal/       FD solver + adjoint
    pyrova/optimizer/     differentiable placer, legalization pass
    pyrova/objectives/    CVaR / DRO / wirelength / overlap terms
    pyrova/workloads/     workload-power models and dataset loaders
    pyrova/evaluation/    estimators and statistics (CVaR, corrected CIs, Holm)
    pyrova/experiments/   one self-contained script per experiment
    pyrova/inputs/        floorplans and thermal configs
    pyrova/results/       experiment outputs
    scripts/              Slurm submission scripts

## Reproduce

Each result file in `pyrova/results/` is produced by exactly one script in
`pyrova/experiments/`; run it directly, or `make reproduce` for the
self-contained set. Long-running studies have Slurm scripts under `scripts/`.

The real-workload experiments need the BOOM dataset (GPL):

    git clone --depth 1 https://github.com/zhaijw18/mcpat-calib-public.git
    export BOOM_DATA=$(pwd)/mcpat-calib-public
