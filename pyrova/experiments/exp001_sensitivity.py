"""Cross-workload sensitivity figure: one mean-objective placement p^j is
optimised per workload xi^j, the full cross-performance matrix
M[i,j] = M(p^i, xi^j) is evaluated, and each placement's worst-case regret
max_j (M[i,j] - M[j,j]) is reported as a heatmap + bar chart. Single seed,
no CIs — a motivation figure, not a hypothesis test.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG  = HERE.parent                               # pyrova
REPO = PKG.parent                                # repo root (has the package)
sys.path.insert(0, str(REPO))

FLP     = PKG / "inputs" / "floorplans" / "ev6.flp"
RESULTS = PKG / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from pyrova.thermal.fd_solver import GridFDSolver, random_power_map
from pyrova.core.design import Design
from pyrova.objectives.thermal import peak_temperature
from pyrova.optimizer.placer import DiffPlacer

N_WORKLOADS = 6        # number of workloads xi^j (= number of placements p^j)
N_ITER      = 40       # Adam iterations per single-workload optimisation
LR          = 1e-2
NR = NC     = 32       # optimisation grid (fast)
TOTAL_P     = 50.0     # W per workload
SEED        = 0


def design_from_positions(base: Design, cx: np.ndarray, cy: np.ndarray) -> Design:
    """Return a copy of `base` with macro centres at (cx, cy)."""
    macros = [m.moved(float(cx[i]) - m.width / 2.0,
                      float(cy[i]) - m.height / 2.0)
              for i, m in enumerate(base.macros)]
    return base.with_macros(macros)


def main() -> None:
    print("=== exp001 - workload sensitivity (cross-performance matrix) ===")
    design = Design.from_flp(str(FLP))
    names  = design.macro_names
    print(f"Design: {design.name}, {len(design.macros)} macros, "
          f"{design.chip_width*1e3:.1f}x{design.chip_height*1e3:.1f} mm")

    # Solver (G depends only on chip geometry + grid, so one solver is reused)
    cfg   = design.thermal_config.as_dict()
    units = design.macro_flp_dicts()
    solver = GridFDSolver(cfg, units, design.chip_width, design.chip_height, NR, NC)
    solver.build(); solver.factorize()
    print(f"Solver N={solver.N}, grid {NR}x{NC}")

    # Workloads xi^j  (dict form for evaluation, array form for the placer)
    rng = np.random.default_rng(SEED)
    xi_dicts  = [random_power_map(units, TOTAL_P, rng) for _ in range(N_WORKLOADS)]
    xi_arrays = [np.array([d[n] for n in names]) for d in xi_dicts]

    # Optimise one placement per workload: p^j = argmin_p M(p, xi^j)
    placements: list[Design] = []
    for j in range(N_WORKLOADS):
        placer = DiffPlacer(solver, units, design.chip_width, design.chip_height,
                            NR, NC, alpha=0.9, eps_dro=0.0)
        placer.optimize([xi_arrays[j]], mode="mean", n_iter=N_ITER, lr=LR,
                        verbose=False)
        cx, cy = placer.get_positions()
        placements.append(design_from_positions(design, cx, cy))
        print(f"  optimised placement p^{j} for workload xi^{j}")

    # Cross-performance matrix M[i, j] = M(p^i, xi^j)
    M = np.zeros((N_WORKLOADS, N_WORKLOADS))
    for i in range(N_WORKLOADS):
        for j in range(N_WORKLOADS):
            M[i, j] = peak_temperature(placements[i], solver, xi_dicts[j])

    # delta_ij = M(p^i, xi^j) - M(p^j, xi^j)   (subtract the column's own-optimum)
    own = np.diag(M).copy()                 # own[j] = M(p^j, xi^j)
    delta = M - own[np.newaxis, :]          # broadcast across rows
    worst_regret = delta.max(axis=1)        # max_j delta_ij per placement i

    print("\ndelta_ij [K] (rows = placement i, cols = workload j):")
    for i in range(N_WORKLOADS):
        print("  " + "  ".join(f"{delta[i, j]:+6.2f}" for j in range(N_WORKLOADS)))
    print("\nWorst-case regret max_j delta_ij per placement:")
    for i in range(N_WORKLOADS):
        print(f"  p^{i}: {worst_regret[i]:.2f} K")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(delta, cmap="inferno", aspect="auto")
    ax.set_xlabel("workload j (xi^j)")
    ax.set_ylabel("placement i (p^i)")
    ax.set_title("Cross-performance delta_ij = M(p^i,xi^j) - M(p^j,xi^j) [K]")
    ax.set_xticks(range(N_WORKLOADS)); ax.set_yticks(range(N_WORKLOADS))
    for i in range(N_WORKLOADS):
        for j in range(N_WORKLOADS):
            ax.text(j, i, f"{delta[i, j]:.1f}", ha="center", va="center",
                    color="white" if delta[i, j] < delta.max() * 0.6 else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dT penalty [K]")

    ax = axes[1]
    ax.bar(range(N_WORKLOADS), worst_regret, color="steelblue")
    ax.set_xlabel("placement i (p^i)")
    ax.set_ylabel("worst-case regret  max_j delta_ij  [K]")
    ax.set_title("Thermal cost of committing to one placement\nacross all workloads")
    ax.set_xticks(range(N_WORKLOADS))

    fig.tight_layout()
    png = RESULTS / "exp001_sensitivity.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {png}")

    txt = RESULTS / "exp001_sensitivity.txt"
    with open(txt, "w") as f:
        f.write("exp001 - workload sensitivity (cross-performance)\n")
        f.write("=" * 55 + "\n")
        f.write(f"N_WORKLOADS={N_WORKLOADS}, N_ITER={N_ITER}, grid {NR}x{NC}, "
                f"TOTAL_P={TOTAL_P} W, seed={SEED}\n")
        f.write("All values are peak delta_T = T_peak - T_ambient [K].\n\n")
        f.write("M[i,j] = M(p^i, xi^j):\n")
        for i in range(N_WORKLOADS):
            f.write("  " + "  ".join(f"{M[i, j]:6.2f}" for j in range(N_WORKLOADS)) + "\n")
        f.write("\nDelta_ij = M(p^i,xi^j) - M(p^j,xi^j):\n")
        for i in range(N_WORKLOADS):
            f.write("  " + "  ".join(f"{delta[i, j]:+6.2f}" for j in range(N_WORKLOADS)) + "\n")
        f.write("\nWorst-case regret max_j Delta_ij per placement:\n")
        for i in range(N_WORKLOADS):
            f.write(f"  p^{i}: {worst_regret[i]:.2f} K\n")
    print(f"Summary -> {txt}")


if __name__ == "__main__":
    main()
