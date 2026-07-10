"""DRO teeth check + i.i.d. negative control: does the penalty run and move OOS CVaR?

Broad, CI-free sanity sweep over every loadable 2D-monolithic benchmark on the i.i.d.
synthetic workload. Two roles: (1) confirm the DRO term has teeth (gradient distinct
from CVaR, optimisation descends the penalty) on each benchmark; (2) the i.i.d.
NEGATIVE CONTROL for the Wasserstein penalty — on i.i.d. there is no tail dimension
(exp004), so DRO is not expected to beat the mean placement. It already reports two
reference points side by side (vs_mean and vs_cvar0), which is the de-confounded
framing; it is deliberately NOT a hypothesis verdict (no CIs, coarse grid, eps
uncalibrated). The matched-CI structured-vs-iid DRO test is exp007.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG = HERE.parent                               # pyrova
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config, random_power_map
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import cvar

CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.9
EPS_SWEEP = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]   # 0.0 = pure CVaR baseline
N_SEEDS = 4
N_TRAIN, N_TEST = 24, 80
NR = NC = 14
N_ITER = 20
MIN_BLOCKS, MAX_BLOCKS = 2, 60   # placement-scale; excludes gridded power maps


# Our solver is a 2D monolithic die (Si/TIM/SP/HS stack). Exclude 3D-IC,
# liquid-cooled, interposer, PDN and gridded-power-map floorplans — they are not
# single-layer 2D placement benchmarks even when they parse.
NON_2D = ("3d", "m3d", "tsv", "tim", "interposer", "microchannel", "liquid",
          "dram", "metal", "tier", "pdn", "scc", "silicon", "heater", "horizontal")
NON_2D_DIRS = ("example4", "example5")   # HotSpot 3D-stack example suites


def _is_2d_monolithic(path: Path) -> bool:
    s = str(path).lower()
    if any(d in s for d in NON_2D_DIRS):
        return False
    return not any(tok in path.stem.lower() for tok in NON_2D)


def discover() -> list[Path]:
    """Loadable 2D-monolithic .flp files with a placement-scale block count."""
    cands = [PKG / "inputs/floorplans/ev6.flp", *sorted((ROOT / "Tools").rglob("*.flp"))]
    seen, keep = set(), []
    for p in cands:
        if not _is_2d_monolithic(p):
            continue
        try:
            units = parse_flp(str(p))
        except Exception:
            continue
        n = len(units)
        if not (MIN_BLOCKS <= n <= MAX_BLOCKS):
            continue
        w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
        h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
        if w <= 0 or h <= 0:
            continue
        key = (n, round(w, 9), round(h, 9))      # dedupe identical floorplans
        if key in seen:
            continue
        seen.add(key)
        keep.append(p)
    return keep


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def oos_cvar(pl: DiffPlacer, test_scen) -> float:
    cx, cy = pl.get_positions()
    peaks = pl._scenario_peaks(cx, cy, test_scen)
    return cvar(peaks, ALPHA)


def scen_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def teeth_check(solver, units, chip_w, chip_h, train) -> tuple[bool, bool]:
    """Isolate the DRO term (no overlap penalty): gradient distinct from CVaR, and
    DRO optimisation lowers the penalty. Returns (works, lever)."""
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC,
                    alpha=ALPHA, eps_dro=0.5, nonoverlap_w=0.0)
    _, gc, _ = pl.objective_and_grad(train, mode="cvar")
    _, gd, _ = pl.objective_and_grad(train, mode="dro")
    grad_distinct = float(np.abs(gd - gc).max())
    pen0 = pl.dro_term(train)
    pl.optimize(train, mode="dro", n_iter=12, lr=2e-2, verbose=False)
    pen1 = pl.dro_term(train)
    return (grad_distinct > 1e-6 and pen0 > 0), (pen1 < 0.999 * pen0)


def evaluate(path: Path, cfg) -> dict:
    units = parse_flp(str(path))
    n = len(units)
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * n

    works = lever = None
    cvar_mean, lam_mean = [], []
    cvar_eps = {e: [] for e in EPS_SWEEP}
    lam_eps = {e: [] for e in EPS_SWEEP}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        train = scen_set(units, tot, rng, N_TRAIN)
        test = scen_set(units, tot, rng, N_TEST)
        if seed == 0:
            works, lever = teeth_check(solver, units, chip_w, chip_h, train)

        det = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA, eps_dro=0.0)
        det.optimize(train, mode="mean", n_iter=N_ITER, lr=2e-2, verbose=False)
        cvar_mean.append(oos_cvar(det, test))
        lam_mean.append(det.tail_sensitivity(train))

        for e in EPS_SWEEP:
            pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA, eps_dro=e)
            pl.optimize(train, mode="dro", n_iter=N_ITER, lr=2e-2, verbose=False)
            cvar_eps[e].append(oos_cvar(pl, test))
            lam_eps[e].append(pl.tail_sensitivity(train))

    mean = lambda xs: float(np.mean(xs))
    return dict(
        name=path.stem, n=n, works=works, lever=lever,
        cvar_mean=mean(cvar_mean), lam_mean=mean(lam_mean),
        cvar_eps={e: mean(cvar_eps[e]) for e in EPS_SWEEP},
        lam_eps={e: mean(lam_eps[e]) for e in EPS_SWEEP},
    )


def report(r, write=None) -> None:
    """Print (and optionally write) one benchmark's eps-sweep table."""
    def line(s):
        print(s)
        if write is not None:
            write.write(s + "\n")

    cm, lm = r["cvar_mean"], r["lam_mean"]
    cvar0 = r["cvar_eps"][0.0]                       # pure-CVaR baseline
    line(f"\n=== {r['name']} ({r['n']} blocks)  [DRO works: "
         f"{'yes' if r['works'] else 'NO'}, leverage: {'yes' if r['lever'] else 'no'}] ===")
    hdr = (f"  {'placement':<14}{'OOS CVaR':>9}  {'Lambda':>7}  {'penalty':>8}  "
           f"{'pen/CVaR':>8}  {'vs_mean':>8}  {'vs_cvar0':>8}")
    line(hdr); line("  " + "-" * (len(hdr) - 2))
    line(f"  {'mean':<14}{cm:>9.3f}  {lm:>7.2f}  {'-':>8}  {'-':>8}  {'+0.000':>8}  {'-':>8}")
    for e in EPS_SWEEP:
        c, lam = r["cvar_eps"][e], r["lam_eps"][e]
        pen = e / (1.0 - ALPHA) * lam
        tag = "cvar(e=0)" if e == 0.0 else f"dro e={e:g}"
        line(f"  {tag:<14}{c:>9.3f}  {lam:>7.2f}  {pen:>8.3f}  {pen/c*100:>7.1f}%  "
             f"{cm - c:>+8.3f}  {cvar0 - c:>+8.3f}")


def main():
    cfg = parse_config(str(CONFIG))
    benches = discover()
    print(f"Found {len(benches)} 2D-monolithic benchmarks "
          f"(blocks in [{MIN_BLOCKS},{MAX_BLOCKS}]):")
    for p in benches:
        print(f"  {p.relative_to(ROOT)}")
    print(f"\neps-sweep, out-of-sample CVaR averaged over {N_SEEDS} seeds "
          f"(alpha={ALPHA}, grid {NR}x{NC}, {N_ITER} iter, synthetic workload).")
    print("vs_mean>0: beats mean placement.  vs_cvar0>0: DRO beats pure CVaR (the primary test).")

    out = PKG / "results/eval_dro_benchmarks.txt"
    with open(out, "w") as f:
        f.write(f"eps-sweep OOS CVaR, {N_SEEDS} seeds, alpha={ALPHA}, grid {NR}x{NC}, "
                f"{N_ITER} iter, synthetic workload. NOT a hypothesis verdict: no CIs, "
                f"coarse grid, eps uncalibrated.\n")
        for p in benches:
            try:
                r = evaluate(p, cfg)
            except Exception as e:
                print(f"{p.stem}: FAILED {type(e).__name__}: {e}")
                continue
            report(r, write=f)
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
