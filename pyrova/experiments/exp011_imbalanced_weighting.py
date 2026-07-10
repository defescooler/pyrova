"""exp011: does IMBALANCED workload weighting rescue the exp009 null?

Every prior experiment weights the N scenarios uniformly (1/N) in the empirical
CVaR. Real deployments do not: usage telemetry is heavily concentrated. This
experiment reruns exp009's mean-opt vs CVaR-opt comparison on the same 80 BOOM
benchmarks and the same synthesised geometry, changing ONLY the scenario
weighting, with weights sourced from public deployment data.

PRE-REGISTERED PREDICTION (verbatim from the experiment brief):

  Using the BOOM workload set from exp009 (80 real RISC-V benchmarks with
  per-functional-unit McPAT power) and real component areas, under three
  independently-sourced imbalanced weighting schemes documented below, we
  predict:

    (a) The imbalanced-weighted anti-correlated cluster structure survives:
        weighted-corr(FP, INT) < -0.1 in at least 2 of 3 weighting schemes.
    (b) Under at least one weighting scheme with concentration ratio > 0.7
        (Gini > 0.5 or similar), dCVaR > 0 with 95% CI strictly above zero.
    (c) The i.i.d. control (uniform weighting) reproduces exp009's null
        (dCVaR CI includes zero) — this validates that the pipeline works
        correctly under the known-null condition.

  Falsification: if all three weighting schemes yield dCVaR CI including zero,
  the imbalanced-weighting mechanism does not rescue the empirical result on
  this design, and we accept the null on real workloads with real deployment
  weightings.

WEIGHTING SCHEMES — sources (all category-level numbers verified against the
primary PDFs; benchmark-level allocation is UNIFORM WITHIN CATEGORY by the
pre-declared rule below — the sources say nothing about individual kernels):

  A "borg2019-dc": Tirmazi et al., "Borg: the Next Generation", EuroSys 2020,
    DOI 10.1145/3342195.3387517. In-text (Sec. 4): best-effort batch ~= 20% of
    cell capacity; production ~= 0.30 of capacity approx from Fig. 2b/3 of
    ~0.60 total usage -> usage shares prod 0.50 / batch 0.33 / mid 0.13 /
    free 0.04. Tier->category: prod -> {mem_stream, data_proc, general};
    batch -> {media_dsp, fp_sci}; mid -> control; free -> trivial.
    NOTE: fails the concentration gate by construction (top10=0.25) — reported
    as a sourced low-concentration arm; prediction (b) is tested on B/C.

  B "mobile-falaki": Falaki et al., "Diversity in Smartphone Usage", MobiSys
    2010, DOI 10.1145/1814433.1814453, Fig. 13 Dataset2 (N=222) shares of
    ACTIVE interaction time: communication 49%, browsing 12%, games 10%,
    media 9%, productivity 2%, maps 2%, system 1%, other 15%. Idle fraction
    1 - 59.23/1440 = 0.9589 from Boehmer et al., MobileHCI 2011, DOI
    10.1145/2037373.2037383 (59.23 min/day mean active use). Activity->category:
    comm->control, browsing+productivity->data_proc, media->media_dsp,
    games->fp_sci, maps->mem_stream, system->general, other->uniform over the
    six app categories, idle->trivial.

  C "mobile-carroll": Carroll & Heiser, "An Analysis of Power Consumption in a
    Smartphone", USENIX ATC 2010, Table 12 "Regular" daily pattern: SMS 30 +
    audio 60 + call 30 + web 15 + email 15 min/day, remainder suspend (89.6%).
    Activity->category: SMS+call->control, audio->media_dsp, web+email->
    data_proc, suspend->trivial.

  (The brief's third suggestion — SPEC frequency weightings — was DROPPED: no
  published survey gives cycle-share frequencies over SPEC-like patterns;
  the SPEC subsetting literature gives redundancy clusters, not deployment
  frequencies. Per the brief's drop rule, C substitutes a second
  independently-sourced mobile scheme.)

  "uniform-control": 1/80 each, exp009's setup exactly (same split RNG).

PRE-DECLARED benchmark->category rule (mechanical, by name; fixed before any
dCVaR was computed): see CATEGORY below. ISA unit tests (rv64*) -> verification,
weight 0 in all deployment schemes (deployed chips do not run ISA loops).
hello_world* -> trivial, used as the idle proxies (the brief forbids adding new
workload models). POST-RUN FINDING (kept, not tuned away, per the brief's
no-post-hoc-adjustment rule): the proxies turn out to be thermally INVALID as
idle — bare-metal hello_world runs the full boot/print path at 6.07/6.29 W vs
5.11 W median (peak-dT ranks 30 and 26 of 80). B and C therefore represent
"weight concentrated on two above-median-power scenarios", NOT idle-dominated
deployments. This is reported in the diagnostics and bounds the conclusion.

DESIGN NOTES (declared before running):
  - Disjoint 40/40 train/test, 10 splits, 95% t-CIs, exactly as exp009.
  - Weighted arms stratify the two `trivial` benchmarks (one per half) so the
    dominant idle atom is represented on both sides; weights renormalised
    within each half. uniform-control uses exp009's plain permutation.
  - Mechanistic prior, recorded up front: exp009 found the hotspot stable in
    76/80 programs and the anti-correlation in the thermally-light FP cluster
    (max FP power share across all 80 benchmarks = 16.6%). Reweighting changes
    which scenarios matter, not the geometry, so the null may survive; that is
    prediction (b)'s falsification branch, and it is an acceptable answer.
  - Under B the active mass (0.041) < tail mass (0.1), so the weighted tail
    includes part of the idle atom (dilution); under C the active mass (0.104)
    fills the tail almost exactly. Reported in the tail diagnostic.
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_nadeau_bengio
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

CONFIG = PKG / "inputs/configs/thermal.config"
CONFIG_ID = "0"          # exp009-matching
ALPHA = 0.90
NR = NC = 18
N_ITER = 30
N_SPLITS = 10
TRAIN_FRAC = 0.5
TARGET_PEAK = 40.0

# Pre-declared benchmark -> category map (mechanical, by name)
_CATS = {
    "media_dsp": ["adpcm_dec", "adpcm_enc", "h264_dec", "huff_dec", "jfdctint",
                  "fir2dim", "iir", "lms", "fmref", "edn", "compress", "crc", "ndes"],
    "fp_sci":    ["sqrt", "fft", "frac", "qurt", "whetstone", "add_fp",
                  "add_fp_large", "expint", "ludcmp", "minver", "basicmath_small",
                  "complex_updates"],
    "data_proc": ["qsort", "rsort", "bsort", "bubblesort", "insertsort",
                  "mergesort", "bitonic", "binarysearch", "levenshtein",
                  "select", "median"],
    "control":   ["branchy", "branchy_large", "cover", "duff", "prime",
                  "fibonacci", "fac", "recursion", "tarai", "towers",
                  "petrinet", "lift", "cnt", "countnegative"],
    "mem_stream": ["vvadd", "spmv", "matrix_mult", "multiply", "st",
                   "add_int", "add_int_large", "bitcount"],
    "general":   ["dhrystone"],
    "trivial":   ["hello_world", "hello_world_large"],
}
CATEGORY = {n: c for c, ns in _CATS.items() for n in ns}

APPCATS = ["media_dsp", "fp_sci", "data_proc", "control", "mem_stream", "general"]


def category_of(name: str) -> str:
    if name.startswith("rv64"):
        return "verification"
    return CATEGORY.get(name, "general")


# Weighting schemes (sourced category weights; see module docstring)
def scheme_weights(names: list[str]) -> dict[str, np.ndarray]:
    cats = [category_of(n) for n in names]

    def from_catw(catw: dict) -> np.ndarray:
        w = np.zeros(len(names))
        for c, cw in catw.items():
            idx = [i for i, cc in enumerate(cats) if cc == c]
            if idx and cw > 0:
                w[idx] = cw / len(idx)
        return w / w.sum()

    # A. Borg 2019 usage shares (prod .50 / batch .33 / mid .13 / free .04)
    borg = {"mem_stream": 0.50 * 8 / 20, "data_proc": 0.50 * 11 / 20,
            "general": 0.50 * 1 / 20,
            "media_dsp": 0.33 * 13 / 24, "fp_sci": 0.33 * 11 / 24,
            "control": 0.13, "trivial": 0.04}

    # B. Falaki Fig.13 Dataset2 active shares + Boehmer idle fraction
    idle = 1.0 - 59.23 / 1440.0
    act = {"control": 0.49, "data_proc": 0.12 + 0.02, "media_dsp": 0.09,
           "fp_sci": 0.10, "mem_stream": 0.02, "general": 0.01}
    act = {k: v + 0.15 / len(APPCATS) for k, v in act.items()}   # spread 'other'
    s = sum(act.values())
    falaki = {k: (1.0 - idle) * v / s for k, v in act.items()}
    falaki["trivial"] = idle

    # C. Carroll Table 12 'Regular' minutes/day
    tot = 1440.0
    carroll = {"control": 60 / tot, "media_dsp": 60 / tot,
               "data_proc": 30 / tot, "trivial": (tot - 150) / tot}

    uniform = np.full(len(names), 1.0 / len(names))
    return {"uniform-control": uniform,
            "A borg2019-dc": from_catw(borg),
            "B mobile-falaki": from_catw(falaki),
            "C mobile-carroll": from_catw(carroll)}


def gini(w: np.ndarray) -> float:
    ws = np.sort(w); n = len(ws)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(ws) / (n * ws.sum()))


def wcorr(a, b, w) -> float:
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return float(cov / np.sqrt(np.average((a - ma) ** 2, weights=w) *
                               np.average((b - mb) ** 2, weights=w)))


def split_indices(n: int, cut: int, seed: int, trivial_idx: list[int],
                  stratify: bool):
    """exp009's plain permutation; weighted arms additionally guarantee one
    trivial (idle-proxy) benchmark per half (declared variance reduction)."""
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(n))
    tr, te = perm[:cut], perm[cut:]
    if stratify and len(trivial_idx) == 2:
        in_tr = [i for i in trivial_idx if i in tr]
        if len(in_tr) == 2:                     # both in train: move one over
            j = int(rng.integers(len(te)))
            k = tr.index(in_tr[1])
            tr[k], te[j] = te[j], tr[k]
        elif len(in_tr) == 0:                   # both in test: move one over
            in_te = [i for i in trivial_idx if i in te]
            j = int(rng.integers(len(tr)))
            k = te.index(in_te[1])
            te[k], tr[j] = tr[j], te[k]
    return tr, te


def main():
    csvp, rptp = resolve_paths()
    if not csvp:
        print("BOOM_DATA not found. Clone the (GPL-3.0) dataset and retry:\n"
              "  git clone --depth 1 https://github.com/zhaijw18/mcpat-calib-public.git\n"
              "  BOOM_DATA=$(pwd)/mcpat-calib-public python -m pyrova.experiments.exp011_imbalanced_weighting")
        return

    out = PKG / "results/exp011_imbalanced_weighting.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s); fh.write(s + "\n")

    cfg = parse_config(str(CONFIG))
    wl = BoomWorkload(csvp, rptp, config_id=CONFIG_ID)
    rows = [r for r in csv.DictReader(open(csvp)) if r["Config_ID"] == CONFIG_ID]
    names = [r["config-benchmark"].replace("GigaBoomConfig-", "").replace(".riscv", "")
             for r in rows]
    assert len(names) == wl.n_programs

    solver = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
    solver.build(); solver.factorize()

    def peaks_fn(scen):
        p = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(peaks_fn, TARGET_PEAK)

    scen = wl.scenarios()
    n = len(scen); cut = int(n * TRAIN_FRAC)
    schemes = scheme_weights(names)
    trivial_idx = [i for i, nm in enumerate(names) if category_of(nm) == "trivial"]

    fam = np.array(wl.families)
    fp_p = wl.power[:, fam == "FP"].sum(1)
    int_p = wl.power[:, fam == "INT"].sum(1)
    mem_p = wl.power[:, fam == "MEM"].sum(1)

    emit(f"exp011: imbalanced deployment weighting on the exp009 BOOM setup "
         f"({n} programs, {len(wl.leaves)} blocks, config {CONFIG_ID}, grid {NR}x{NC}, "
         f"alpha={ALPHA}, {N_SPLITS} splits of {cut}/{n-cut}).")
    emit("Weights: category-level from published telemetry (see script docstring), "
         "uniform within category; rv64* ISA tests weight 0 in deployment schemes.")

    # Phase-1 table (concentration + weighted structure)
    emit("\n(1) Scheme statistics (gate: concentration top10>0.7 or gini>0.5):")
    base_peaks = peaks_fn(scen)
    # Idle-proxy thermal validity (measured, not assumed):
    for nm in ("hello_world", "hello_world_large"):
        i = names.index(nm)
        r = int((base_peaks > base_peaks[i]).sum())
        emit(f"  idle-proxy check: {nm} = {wl.power[i].sum():.2f} W "
             f"(median {np.median(wl.power.sum(1)):.2f} W), peak-dT rank {r}/{n} "
             f"-> {'INVALID as idle proxy (above-median power)' if r < n//2 else 'plausible idle proxy'}")
    stats = {}
    tail_reduced = {}          # arms whose tail trips the >0.3 single-benchmark rule
    for label, w in schemes.items():
        t10 = float(np.sort(w)[-10:].sum()); g = gini(w)
        cFI = wcorr(fp_p, int_p, w); cFM = wcorr(fp_p, mem_p, w)
        stats[label] = dict(t10=t10, gini=g, cFI=cFI, cFM=cFM)
        # weighted-tail composition at the original placement (diagnostic)
        order = np.argsort(-base_peaks)
        ws = w[order]; m = 1.0 - ALPHA
        cum = np.concatenate(([0.0], np.cumsum(ws)))
        phi = np.clip(m - cum[:-1], 0.0, ws)
        members = [(names[order[i]], float(phi[i])) for i in range(n) if phi[i] > 1e-9]
        big = [nm for nm, p in members if p / m > 0.3]
        tail_reduced[label] = big
        emit(f"  {label:18s} top10={t10:.3f} gini={g:.3f} "
             f"w-corr(FP,INT)={cFI:+.3f} w-corr(FP,MEM)={cFM:+.3f}")
        emit(f"    tail@alpha={ALPHA} ({len(members)} members): "
             + ", ".join(f"{nm}:{p/m:.2f}" for nm, p in members[:6])
             + (" ..." if len(members) > 6 else "")
             + (f"  [WARNING: single-benchmark tail mass >0.3: {big}]" if big else ""))

    # Placement arms
    emit(f"\n(2) PLACEMENT: dCVaR/dMean = (mean-opt) - (cvar-opt), weighted OOS, "
         f"{N_SPLITS} splits, 95% t-CI:")
    results = {}
    for label, w in schemes.items():
        stratify = label != "uniform-control"
        dC, dM = [], []
        for seed in range(N_SPLITS):
            tr_i, te_i = split_indices(n, cut, seed, trivial_idx, stratify)
            tr = [scen[i] for i in tr_i]; te = [scen[i] for i in te_i]
            w_tr = w[tr_i] / w[tr_i].sum() if stratify else None
            w_te = w[te_i] / w[te_i].sum() if stratify else None

            pm = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
            pm.optimize(tr, mode="mean", n_iter=N_ITER, lr=2e-2, verbose=False,
                        weights=w_tr)
            pc = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
            pc.optimize(tr, mode="cvar", n_iter=N_ITER, lr=2e-2, verbose=False,
                        weights=w_tr)

            cxm, cym = pm.get_positions()
            mm, cm = mean_cvar(pm._scenario_peaks(cxm, cym, te), ALPHA, weights=w_te)
            cxc, cyc = pc.get_positions()
            mc, cc = mean_cvar(pc._scenario_peaks(cxc, cyc, te), ALPHA, weights=w_te)
            dC.append(cm - cc); dM.append(mm - mc)
        # Repeated random splits of one fixed pool share data across splits:
        # NB-corrected CI (SE = s*sqrt(1/J + n_te/n_tr), as exp009), not the naive t.
        ratio = (n - cut) / cut
        gC, _, lo, hi = ci95_nadeau_bengio(dC, ratio)
        gM, _, mlo, mhi = ci95_nadeau_bengio(dM, ratio)
        flag = "*" if lo > 0 else ("x" if hi < 0 else "ns")
        results[label] = dict(dC=gC, lo=lo, hi=hi, dM=gM, mlo=mlo, mhi=mhi, flag=flag)
        emit(f"  {label:18s} dCVaR={gC:+.3f} K CI[{lo:+.3f},{hi:+.3f}] {flag:2s}  "
             f"dMean={gM:+.3f} K CI[{mlo:+.3f},{mhi:+.3f}]")

    # Verdict vs pre-registration
    emit("\n(3) VERDICT vs the pre-registered prediction:")
    n_anti = sum(1 for lb in ("A borg2019-dc", "B mobile-falaki", "C mobile-carroll")
                 if stats[lb]["cFI"] < -0.1)
    emit(f"  (a) weighted-corr(FP,INT) < -0.1 in {n_anti}/3 schemes -> "
         f"{'CONFIRMED' if n_anti >= 2 else 'FALSIFIED'}")
    eligible = [lb for lb in ("A borg2019-dc", "B mobile-falaki", "C mobile-carroll")
                if stats[lb]["t10"] > 0.7 or stats[lb]["gini"] > 0.5]
    b_pos = [lb for lb in eligible if results[lb]["flag"] == "*"]
    b_clean = [lb for lb in b_pos if not tail_reduced.get(lb)]
    emit(f"  (b) concentrated schemes {eligible}; dCVaR CI>0 in {b_pos or 'none'} -> "
         f"{'CONFIRMED' if b_pos else 'FALSIFIED'}"
         + ("" if b_pos == b_clean else
            f" — BUT the pre-registered tail diagnostic fired on {[lb for lb in b_pos if tail_reduced.get(lb)]}: "
            f"per the brief's reduction rule, those positives reduce to 'CVaR-opt "
            f"optimizes for the specific high-weight tail benchmarks'."))
    uc = results["uniform-control"]
    c_ok = uc["lo"] <= 0 <= uc["hi"]
    emit(f"  (c) uniform-control dCVaR={uc['dC']:+.3f} CI[{uc['lo']:+.3f},{uc['hi']:+.3f}] "
         f"-> {'null reproduced (pipeline valid)' if c_ok else 'UNEXPECTEDLY SIGNIFICANT - pipeline problem'}")

    # dMean pattern check against the de-confounded doctrine: dCVaR>0 AND dMean<0
    # is a mean-for-tail trade; dCVaR>0 AND dMean>0 means mean-opt is DOMINATED,
    # i.e. the weighted-mean objective generalised badly (coverage effect), not
    # risk aversion buying tail at mean cost.
    dominated = [lb for lb in results if lb != "uniform-control"
                 and results[lb]["lo"] > 0 and results[lb]["mlo"] > 0]
    if dominated:
        emit(f"  NOTE: in {dominated} CVaR-opt beats mean-opt on BOTH weighted mean and "
             f"weighted CVaR (dMean CI>0). That is domination, not the theory's "
             f"mean-for-tail trade: concentrated weights collapse the weighted-mean "
             f"objective toward few scenarios and it generalises badly; the CVaR "
             f"objective keeps multi-scenario coverage.")

    if not c_ok:
        overall = "AMBIGUOUS: uniform-control unexpectedly significant; investigate before reading the arms."
    elif b_clean:
        overall = ("CONFIRMED on " + ", ".join(b_clean) + ": under sourced deployment weightings, "
                   "risk-aware placement shows a significant tail benefit on real workloads.")
    elif b_pos:
        overall = ("CONFIRMED-WITH-REDUCTION: the CI criterion passed on " + ", ".join(b_pos) +
                   ", but every confirming arm trips the pre-registered tail-domination "
                   "diagnostic, and the idle proxy is thermally invalid (above-median power). "
                   "Defensible claim: when deployment weight concentrates on a few scenarios, "
                   "weighted-CVaR placement beats weighted-mean placement OOS — a scenario-"
                   "coverage effect (CVaR-opt dominates BOTH metrics), decoupled from the "
                   "anti-correlation mechanism (largest positive occurs in scheme C where the "
                   "weighted structure is destroyed, w-corr(FP,INT)=-0.05). NOT the theory's "
                   "mean-for-tail trade; NOT evidence the exp009 mechanism-null is wrong.")
    else:
        overall = ("FALSIFIED: the exp009 null persists under sourced deployment weightings; "
                   "the geometric mechanism (thermally-light FP cluster, stable hotspot) "
                   "dominates the weighting effect on this design.")
    emit(f"  OVERALL: {overall}")
    emit("\nCaveats: geometry synthesised from McPAT areas (exp009 caveat carries over); "
         "benchmark-level weights are category-sourced + uniform-within-category, not "
         "measured benchmark telemetry; hello_world* used as idle proxies turned out "
         "to be ABOVE-median power (see idle-proxy check above) — B/C represent "
         "weight concentration, not idle-dominated deployments; a true near-zero "
         "idle scenario would need a new pre-registered run.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
