# Pyrova — single auditable entry point from a clean checkout to results.
# Every step from ground-truth data to artifact runs from a target here,
# so results are reproducible without ad-hoc commands.

PYTHON ?= python
RESULTS := pyrova/results

.PHONY: help install verify golden gradients test reproduce reproduce-real clean

help:
	@echo "make install    - editable install into the active environment"
	@echo "make verify     - golden field snapshot + adjoint/placer gradient checks"
	@echo "make golden      - regenerate the golden reference (only after a vetted change)"
	@echo "make reproduce  - run every self-contained experiment into $(RESULTS)/"
	@echo "make clean       - remove caches and generated results"

install:
	$(PYTHON) -m pip install -e .

# The verification gate. Both checks must pass before any numeric change lands.
verify: golden-check gradients

golden-check:
	$(PYTHON) -m pyrova.tests.golden

gradients:
	$(PYTHON) -m pyrova.tests.test_gradients

test: verify

golden:
	$(PYTHON) -m pyrova.tests.golden --write

# Experiments are single-file, ground-truth-driven scripts. The real-workload
# experiments need the external GPL BOOM dataset: set BOOM_DATA to a clone of
# mcpat-calib-public.
reproduce:
	@mkdir -p $(RESULTS)
	$(PYTHON) pyrova/experiments/exp010_blend_objective.py
	$(PYTHON) pyrova/experiments/exp005_structured_workload.py
	$(PYTHON) pyrova/experiments/exp007_structured_dro.py
	$(PYTHON) pyrova/experiments/exp008_correlation_threshold.py
	$(PYTHON) pyrova/experiments/exp004_tail_sample_sweep.py
	$(PYTHON) pyrova/experiments/exp006_real_traces.py
	$(PYTHON) pyrova/experiments/exp003_mean_cvar_correlation.py
	$(PYTHON) pyrova/experiments/eval_dro_benchmarks.py
	$(PYTHON) pyrova/experiments/exp001_sensitivity.py

reproduce-real:
	@mkdir -p $(RESULTS)
	$(PYTHON) pyrova/experiments/exp009_boom_real_traces.py
	$(PYTHON) pyrova/experiments/exp011_imbalanced_weighting.py

clean:
	rm -rf $(RESULTS)/*.txt $(RESULTS)/*.png
	find pyrova -name __pycache__ -type d -prune -exec rm -rf {} +
