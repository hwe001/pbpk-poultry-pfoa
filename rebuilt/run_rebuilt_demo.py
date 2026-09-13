#!/usr/bin/env python3
"""Demonstration of the rebuilt poultry PFOA PBPK model.

Produces, for each species:
  1. Time courses under the legacy 7-day scenario (daily boluses, 0-700 h);
  2. The 700-h liver/muscle/egg concentrations vs. the Chongqing calibration
     targets (Tang et al. 2024) AS CURRENTLY TRANSCRIBED -- i.e. in ug/L,
     a unit basis the co-authors must still confirm against the source;
  3. A preliminary reduced refit (k_u, P_liver, P_muscle, P_egg) to show the
     calibration machinery works on the rebuilt structure.

STATUS CAVEATS (do not strip these when reusing outputs):
  - The physiological volumes/flows carry ASSUMPTION flags (see model docstring)
    and several need verification against Wang et al. (2021).
  - Target units are unconfirmed; the fit is in-sample against 3 means with 4
    free parameters, so it demonstrates machinery, NOT predictive performance.
  - Nothing here is a publication result until steps 2-5 of the rebuild plan
    (unit confirmation, recalibration, validation, rerun of MC/Sobol) are done.

Run:  python run_rebuilt_demo.py     (writes results/ figures + CSV)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from pbpk_poultry_rebuilt import (I_EGG, I_LIVER, I_MUSCLE, VOLUMES,
                                  chicken_params, duck_params, legacy_scenario,
                                  simulate, SpeciesParams)

OUT = Path(__file__).resolve().parents[1] / "results"

# Calibration targets as transcribed from Tang et al. (2024) via the legacy
# manuscript -- UNIT BASIS UNCONFIRMED (manuscript text is ug/L; tissue
# conventions are usually ug/kg wet weight; verify before any refit used in
# the paper).
TARGETS = {
    "chicken": {"liver": 0.251, "muscle": 0.141, "egg": 0.128},
    "duck": {"liver": 0.173, "muscle": 0.210, "egg": 0.220},
}
TISSUE_IDX = {"liver": I_LIVER, "muscle": I_MUSCLE, "egg": I_EGG}


def conc_at(sol, tissue: str, t_query: float = 700.0) -> float:
    i = TISSUE_IDX[tissue]
    amounts = sol.amounts[i] / VOLUMES[tissue]
    return float(np.interp(t_query, sol.t, amounts))


def refit_reduced(params: SpeciesParams):
    """Preliminary refit of {k_u, P_liver, P_muscle, P_egg} against the 3 targets.

    The remaining partition coefficients (kidney, fat, rest) and ka are held at
    their initial values: they are unidentifiable against these 3 targets, and
    a 4-parameter fit to 3 points is already in-sample calibration, nothing more.
    """
    tgt = TARGETS[params.name]
    names = ["liver", "muscle", "egg"]
    x0 = np.array([params.k_u, params.P["liver"], params.P["muscle"], params.P["egg"]])
    lo = np.array([1e-5, 0.1, 0.1, 0.1])
    hi = np.array([1.0, 50.0, 50.0, 50.0])

    def residual(x):
        trial = params.with_overrides(
            k_u=x[0], P={**params.P, "liver": x[1], "muscle": x[2], "egg": x[3]})
        sol = legacy_scenario(trial)
        return np.array([np.log10(max(conc_at(sol, n), 1e-12) / tgt[n]) for n in names])

    res = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
    fitted = res.x
    fitted_params = params.with_overrides(
        k_u=fitted[0], P={**params.P, "liver": fitted[1],
                          "muscle": fitted[2], "egg": fitted[3]})
    return fitted_params, res


def report(sol_before, sol_after, fitted_params, res) -> None:
    p = fitted_params
    print(f"\n--- {p.name}: 700-h concentrations (ug/L, unit basis UNCONFIRMED) ---")
    print(f"{'tissue':>8s} {'target':>9s} {'before fit':>11s} {'after fit':>10s}")
    for n in ("liver", "muscle", "egg"):
        print(f"{n:>8s} {TARGETS[p.name][n]:>9.3f} {conc_at(sol_before, n):>11.3f} "
              f"{conc_at(sol_after, n):>10.3f}")
    print(f"fitted: k_u={p.k_u:.4g} h^-1, P_liver={p.P['liver']:.3g}, "
          f"P_muscle={p.P['muscle']:.3g}, P_egg={p.P['egg']:.3g}  "
          f"(cost={res.cost:.3g})")
    print(f"mass-balance error: before={sol_before.mass_err:.1e}, "
          f"after={sol_after.mass_err:.1e}")


def plot_species(ax, sols, label_prefix: str) -> None:
    colors = {"liver": "#b23a3a", "muscle": "#2a78d6", "egg": "#c98a1b", "plasma": "#4a8a4a"}
    for tissue in ("liver", "muscle", "egg", "plasma"):
        i = TISSUE_IDX.get(tissue, 1)
        c = sols.amounts[i] / (VOLUMES[tissue] if tissue in VOLUMES else 1.0)
        ax.plot(sols.t, c, color=colors[tissue], lw=1.6,
                label=f"{label_prefix} {tissue}")
    for t_dose in range(0, 145, 24):
        ax.axvline(t_dose, color="grey", lw=0.6, alpha=0.4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    fitted_all = {}
    for ax_row, make_params in ((axes[0], chicken_params), (axes[1], duck_params)):
        p0 = make_params()
        sol0 = legacy_scenario(p0)
        p1, res = refit_reduced(p0)
        sol1 = legacy_scenario(p1)
        report(sol0, sol1, p1, res)
        fitted_all[p0.name] = p1

        ax = ax_row[0]  # initial (legacy) parameter values
        plot_species(ax, sol0, f"{p0.name} (initial)")
        ax.set_title(f"{p0.name} — initial (legacy) parameters", fontsize=11)
        ax = ax_row[1]  # after preliminary reduced refit
        plot_species(ax, sol1, f"{p0.name} (refit)")
        ax.set_title(f"{p0.name} — preliminary reduced refit", fontsize=11)

    for ax in axes.flat:
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (µg/kg wet weight)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Rebuilt flow-limited PBPK model — legacy 7-day scenario diagnostic\n"
                 "(mass-balance verified; see Results for interpretation)", fontsize=11)
    fig.tight_layout()
    fig_path = OUT / "rebuilt_demo_chicken_duck.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\nFigure written to {fig_path}")

    csv_path = OUT / "rebuilt_demo_700h_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("species,parameter,value,note\n")
        for name, p in fitted_all.items():
            f.write(f"{name},k_u,{p.k_u:.6g},h^-1 refit\n")
            for t in ("liver", "kidney", "muscle", "fat", "egg", "rest"):
                f.write(f"{name},P_{t},{p.P[t]:.6g},"
                        f"{'refit' if t in ('liver', 'muscle', 'egg') else 'initial'}\n")
            for t, v in TARGETS[name].items():
                f.write(f"{name},target_{t},{v},ug/L UNCONFIRMED units\n")
    print(f"Summary CSV written to {csv_path}")


if __name__ == "__main__":
    main()
