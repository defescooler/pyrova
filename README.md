# Pyrova

A differentiable thermal-aware macro placer.

## Setup

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

## Run

Engine validation:

    python -m pyrova.tests.test_gradients

Experiments (pyrova/results/):

    python pyrova/experiments/exp001_sensitivity.py            # workload->placement sensitivity
    python pyrova/experiments/exp003_mean_cvar_correlation.py  # mechanism
    python pyrova/experiments/exp004_tail_sample_sweep.py      # N_train x alpha learnability sweep, i.i.d.
    python pyrova/experiments/exp005_structured_workload.py    # exp004 under structured workload 
    python pyrova/experiments/exp006_real_traces.py            # real-trace probe 
    python pyrova/experiments/exp007_structured_dro.py         # DRO vs pure CVaR at small N 
    python pyrova/experiments/eval_dro_benchmarks.py           # i.i.d. DRO teeth + negative control