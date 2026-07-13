# Pyrova — single auditable entry point from a clean checkout to results.
# Every step from ground-truth data to artifact runs from a target here,
# so results are reproducible without ad-hoc commands.

PYTHON ?= python
RESULTS := pyrova/results

.PHONY: help install verify golden golden-check gradients test reproduce reproduce-real clean

help:
	@echo "make install    - editable install into the active environment"
	@echo "make verify     - golden field snapshot + adjoint/placer gradient checks"
	@echo "make golden     - regenerate the golden reference (only after a vetted change)"
	@echo "make reproduce  - run the self-contained laptop-scale experiments into $(RESULTS)/"
	@echo "make clean      - remove caches and generated results"

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

# Laptop-scale, fully self-contained experiments (hours, no external data).
# The cluster-scale studies (exp02x-exp044) run for hours-to-days each and
# are not part of this target; run their scripts directly on a batch system.
reproduce:
	@mkdir -p $(RESULTS)
	$(PYTHON) pyrova/experiments/exp001_sensitivity.py
	$(PYTHON) pyrova/experiments/exp003_mean_cvar_correlation.py
	$(PYTHON) pyrova/experiments/exp004_tail_sample_sweep.py
	$(PYTHON) pyrova/experiments/exp005_structured_workload.py
	$(PYTHON) pyrova/experiments/exp008_correlation_threshold.py
	$(PYTHON) pyrova/experiments/exp010_blend_objective.py

# Real-workload experiments; need the external GPL BOOM dataset:
# set BOOM_DATA to a clone of mcpat-calib-public.
reproduce-real:
	@mkdir -p $(RESULTS)
	$(PYTHON) pyrova/experiments/exp006_real_traces.py
	$(PYTHON) pyrova/experiments/exp009_boom_real_traces.py
	$(PYTHON) pyrova/experiments/exp011_imbalanced_weighting.py

clean:
	rm -rf $(RESULTS)/*.txt $(RESULTS)/*.png
	find pyrova -name __pycache__ -type d -prune -exec rm -rf {} +
