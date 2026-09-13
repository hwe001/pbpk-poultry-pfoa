#!/usr/bin/env python3
"""Reverse dosimetry for the rebuilt poultry PFOA PBPK model.

The demonstration refit (run_rebuilt_demo.py) showed the legacy scenario --
7 daily boluses followed by decay to a 700-h snapshot -- can only match the
Chongqing tissue means with an elimination half-life of 4-7 h, ~2 orders of
magnitude faster than PFOA's known persistence. The scenario, not the model,
was the problem: market-survey birds are continuously exposed, so the right
question is a steady-state one.

This script therefore asks the REVERSE question, which is linear and exact
for this model:

    What constant exposure rate R (ug/kg BW/day) yields steady-state tissue
    concentrations equal to the survey means?

Because the system is linear at steady state, tissue concentrations scale
proportionally with R, so:
  * unit-rate steady state (analytic, exact) gives all predictions;
  * D_required(tissue) = target(tissue) / C_ss(tissue) at unit rate;
  * the spread of D_required across tissues ("pattern consistency factor")
    is a scale-free test of whether ONE exposure scalar can explain the
    observed liver/muscle/egg pattern;
  * all absolute exposures inherit the assumed t_half (D_req is exactly
    proportional to k_u = ln2/t_half), so results are reported as functions
    of the assumed half-life, with the assumption swept rather than hidden;
  * tissue-concentration RATIOS are independent of k_u entirely, and the
    required-exposure estimates are invariant to a uniform rescaling of all
    targets -- so the pattern test is robust to the unresolved µg/L vs
    µg/kg transcription question; only the absolute back-calculated exposure
    inherits it.

Outputs: steady-state verification, per-tissue required exposures, MC
uncertainty band, manual Saltelli/Sobol indices (SALib is not available --
the estimator is implemented here and self-tested against a known function),
husbandry translation to feed/water concentrations, figures and CSV/JSON.

NOT publication results until the calibration-target units are confirmed and
the assumption-flagged parameters are reviewed (see model docstring).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pbpk_poultry_rebuilt import (COMPARTMENTS, I_GUT, N_STATES, Q_CO,
                                  SpeciesParams, chicken_params, duck_params, rhs)

OUT = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(20260913)

# Calibration targets (Tang et al. 2024 as transcribed; UNIT BASIS UNCONFIRMED)
TARGETS = {
    "chicken": {"liver": 0.251, "muscle": 0.141, "egg": 0.128},
    "duck": {"liver": 0.173, "muscle": 0.210, "egg": 0.220},
}

# Husbandry assumptions for translating required dose to feed/water levels
HUSBANDRY = {
    "chicken": {"BW_kg": 2.0, "feed_kg_day": 0.125, "water_L_day": 0.25},  # ASSUMPTION (typical laying hen)
    "duck": {"BW_kg": 2.5, "feed_kg_day": 0.170, "water_L_day": 0.50},     # ASSUMPTION (laying duck, generous water)
}
T_HALF_DEFAULT_DAYS = 5.0   # ASSUMPTION: swept below; no reliable avian value identified
T_HALF_SWEEP = np.array([0.5, 1, 2, 3, 5, 7, 10, 14])


# --- Analytic steady state ----------------------------------------------------

def steady_state(p: SpeciesParams, rate_per_h: float) -> dict[str, float]:
    """Exact steady state under continuous gut input `rate_per_h` (ug/kg BW/h).

    Derivation (all flows Q sum to Q_CO):
      gut:      A_gut = rate/ka, absorbed flux F = rate  (F_abs = 1)
      liver:    C_liv = P_liv*(C_p + F/Q_liv)
      tissue i: C_i   = P_i*C_p          (kidney, muscle, fat, rest)
      egg:      C_egg = P_egg*Q_egg*C_p / (Q_egg + k_ovo*V_egg*P_egg)
      plasma:   C_p   = F / (k_u*V_p + egg_sink),
                egg_sink = Q_egg*k_ovo*V_egg*P_egg / (Q_egg + k_ovo*V_egg*P_egg)
    """
    F = rate_per_h
    Vp, Q, P, V = p.V["plasma"], p.Q, p.P, p.V
    egg_denom = Q["egg"] + p.k_ovo * V["egg"] * P["egg"]
    egg_sink = Q["egg"] * p.k_ovo * V["egg"] * P["egg"] / egg_denom
    cp = F / (p.k_u * Vp + egg_sink)
    c = {name: P[name] * cp for name in ("kidney", "muscle", "fat", "rest")}
    c["liver"] = P["liver"] * (cp + F / Q["liver"])
    c["egg"] = P["egg"] * Q["egg"] * cp / egg_denom
    c["plasma"] = cp
    return c


def check_steady_state_vs_integration(p: SpeciesParams, rate_per_h: float = 1.0 / 24.0,
                                      t_end: float = 6000.0) -> float:
    """Integrate the ODEs to (near) steady state and compare with the analytic
    solution -- guards against a transcription error in the closed form."""
    def rhs_continuous(t, y):
        d = rhs(t, y, p)
        d[I_GUT] += rate_per_h
        return d

    y0 = np.zeros(N_STATES)
    sol = solve_ivp(rhs_continuous, (0, t_end), y0, method="Radau",
                    rtol=1e-8, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    worst = 0.0
    exact = steady_state(p, rate_per_h)
    for name in ("plasma", "liver", "kidney", "muscle", "fat", "egg", "rest"):
        idx = 1 + COMPARTMENTS.index(name) - 1
        c_ode = sol.y[idx, -1] / p.V[name]
        worst = max(worst, abs(c_ode - exact[name]) / exact[name])
    return worst


# --- Unit-rate predictions and required exposure --------------------------------

def unit_predictions(p: SpeciesParams, t_half_days: float) -> dict[str, float]:
    """Steady-state concentrations (ug/L) for 1 ug/kg BW/day continuous exposure."""
    p_t = replace(p, k_u=np.log(2.0) / (t_half_days * 24.0))
    return steady_state(p_t, rate_per_h=1.0 / 24.0)


def required_exposure(p: SpeciesParams, t_half_days: float) -> dict[str, float]:
    """Required constant exposure (ug/kg BW/day) to reach each target tissue."""
    unit = unit_predictions(p, t_half_days)
    return {t: TARGETS[p.name][t] / unit[t] for t in TARGETS[p.name]}


def pattern_consistency(p: SpeciesParams, t_half_days: float) -> float:
    d = required_exposure(p, t_half_days).values()
    return max(d) / min(d)


# --- Monte Carlo uncertainty ----------------------------------------------------

def mc_required_exposure(p: SpeciesParams, n: int = 20000,
                         rng: np.random.Generator = RNG) -> dict[str, np.ndarray]:
    """Lognormal parameter uncertainty propagated to required exposure.

    t_half lognormal (median 5 d, sigma_ln 0.7 -- wide, assumption-swept);
    partition coefficients lognormal (sigma_ln 0.35); oviposition rate
    lognormal (sigma_ln 0.2). D_required is exactly proportional to k_u,
    so the t_half spread passes straight through.
    """
    t_half = rng.lognormal(np.log(T_HALF_DEFAULT_DAYS), 0.7, n)
    p_liv = rng.lognormal(np.log(p.P["liver"]), 0.35, n)
    p_mus = rng.lognormal(np.log(p.P["muscle"]), 0.35, n)
    p_egg = rng.lognormal(np.log(p.P["egg"]), 0.35, n)
    k_ovo = rng.lognormal(np.log(p.k_ovo), 0.2, n)

    kt = np.log(2.0) / (t_half * 24.0)
    egg_denom = p.Q["egg"] + k_ovo * p.V["egg"] * p_egg
    egg_sink = p.Q["egg"] * k_ovo * p.V["egg"] * p_egg / egg_denom
    cp = (1.0 / 24.0) / (kt * p.V["plasma"] + egg_sink)
    c_liv = p_liv * (cp + (1.0 / 24.0) / p.Q["liver"])
    c_mus = p_mus * cp
    c_egg = p_egg * p.Q["egg"] * cp / egg_denom
    tgt = TARGETS[p.name]
    return {"liver": tgt["liver"] / c_liv, "muscle": tgt["muscle"] / c_mus,
            "egg": tgt["egg"] / c_egg, "_t_half": t_half}


# --- Formulation and absorption sensitivity (reviewer request) -------------------

def absorption_sensitivity(p: SpeciesParams, f_values=(1.0, 0.8, 0.6, 0.4)) -> dict:
    """Required exposure if only a fraction F of intake is absorbed.

    Absorbed flux = F * R, and D_required scales exactly as 1/F; computed
    numerically for completeness.
    """
    out = {}
    for F in f_values:
        p_t = replace(p, k_u=np.log(2.0) / (T_HALF_DEFAULT_DAYS * 24.0))
        c = steady_state(p_t, rate_per_h=F / 24.0)
        tgt = TARGETS[p.name]
        d = {t: tgt[t] / c[t] for t in tgt}
        out[F] = float(np.exp(np.mean(np.log(list(d.values())))))
    return out


def formulation_sensitivity(p: SpeciesParams) -> dict:
    """Sensitivity of the back-calculated exposure to the egg-elimination
    formulation, at t_half = 5 d.

    baseline : well-mixed egg compartment with first-order oviposition loss
               (laid eggs are an elimination route) -- the manuscript model.
    no_sink  : egg treated as a passive monitoring tissue; oviposition NOT
               counted as elimination (the implicit legacy treatment).
    """
    F = 1.0 / 24.0
    kt = np.log(2.0) / (T_HALF_DEFAULT_DAYS * 24.0)
    tgt = TARGETS[p.name]

    def d_req(cp, c_egg_per_cp):
        c_liv = p.P["liver"] * (cp + F / p.Q["liver"])
        c_mus = p.P["muscle"] * cp
        c_egg = c_egg_per_cp * cp
        d = {"liver": tgt["liver"] / c_liv, "muscle": tgt["muscle"] / c_mus,
             "egg": tgt["egg"] / c_egg}
        pf = max(d.values()) / min(d.values())
        dh = float(np.exp(np.mean(np.log(list(d.values())))))
        return dh, pf

    ed = p.Q["egg"] + p.k_ovo * p.V["egg"] * p.P["egg"]
    es = p.Q["egg"] * p.k_ovo * p.V["egg"] * p.P["egg"] / ed
    cp_b = F / (kt * p.V["plasma"] + es)
    dh_b, pf_b = d_req(cp_b, p.P["egg"] * p.Q["egg"] / ed)
    cp_n = F / (kt * p.V["plasma"])
    dh_n, pf_n = d_req(cp_n, p.P["egg"])
    return {
        "baseline": {"d_hat": dh_b, "pattern_factor": pf_b},
        "no_sink": {"d_hat": dh_n, "pattern_factor": pf_n},
        "ratio_no_sink_over_baseline": dh_n / dh_b,
    }


# --- Sobol indices via definitional conditional sampling -------------------------
# SALib is not installed here, and the classical Saltelli (2002) cross-sampling
# estimators are notoriously easy to mis-pair. We therefore estimate the
# variance-decomposition indices (Sobol 2001) directly from their definitions:
#   S1_i = Var_{x_i}[ E_{x~i}(f | x_i) ] / Var(f)
#   ST_i = E_{x~i}[ Var_{x_i}(f | x~i) ] / Var(f)
# using Monte Carlo with a common random-number grid, and self-test the
# implementation against analytically known indices for additive and
# interacting test functions below.

def sobol_indices_direct(f, bounds: dict[str, tuple[float, float]],
                         m: int = 256, r: int = 512, n_bg: int = 8192,
                         rng: np.random.Generator = RNG) -> tuple[np.ndarray, np.ndarray]:
    """Definition-based S1/ST. f takes an (..., k) array. m grid points per
    variable, r conditional samples, n_bg background points for Var(f).
    Returns (S1, ST) in bounds-key order."""
    names = list(bounds)
    k = len(names)
    lo = np.array([bounds[x][0] for x in names])
    hi = np.array([bounds[x][1] for x in names])

    var_f = np.var(f(rng.uniform(size=(n_bg, k)) * (hi - lo) + lo), ddof=1)
    X = rng.uniform(size=(r, k)) * (hi - lo) + lo          # background for ST

    grid = rng.uniform(size=(m, k)) * (hi - lo) + lo       # stratification points
    S1 = np.zeros(k)
    ST = np.zeros(k)
    for i in range(k):
        # S1: E_{x~i}(f | x_i) -- fix x_i on the grid, average over random complements
        cond_means = np.empty(m)
        for j, xv in enumerate(grid[:, i]):
            Xc = rng.uniform(size=(r, k)) * (hi - lo) + lo
            Xc[:, i] = xv
            cond_means[j] = f(Xc).mean()
        S1[i] = np.var(cond_means, ddof=1) / var_f

        # ST: Var_{x_i}(f | x~i) -- fix the complement at background rows, vary x_i
        cond_vars = np.empty(r)
        for j in range(r):
            Xc = np.repeat(X[j:j + 1], m, axis=0)
            Xc[:, i] = rng.uniform(lo[i], hi[i], m)
            cond_vars[j] = f(Xc).var(ddof=1)
        ST[i] = cond_vars.mean() / var_f
    return S1, ST


def sobol_self_test() -> None:
    """Exact indices: f=x1+2x2 -> S1=ST=(0.2,0.8); f=x1*x2 -> S1=(3/7,3/7),
    ST=(4/7,4/7). Exercises both additive and interacting cases. Tolerances
    reflect Monte Carlo convergence, not exactness."""
    rng = np.random.default_rng(7)
    b = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    S1, ST = sobol_indices_direct(lambda x: x[:, 0] + 2.0 * x[:, 1], b,
                                  m=1024, r=2048, n_bg=65536, rng=rng)
    assert np.allclose(S1, [0.2, 0.8], atol=0.025), f"additive S1 failed: {S1}"
    assert np.allclose(ST, [0.2, 0.8], atol=0.025), f"additive ST failed: {ST}"
    S1, ST = sobol_indices_direct(lambda x: x[:, 0] * x[:, 1], b,
                                  m=1024, r=2048, n_bg=65536, rng=rng)
    assert np.allclose(S1, [3 / 7, 3 / 7], atol=0.025), f"product S1 failed: {S1}"
    assert np.allclose(ST, [4 / 7, 4 / 7], atol=0.04), f"product ST failed: {ST}"
    print("[PASS] Sobol estimator self-tests (additive and interacting functions "
          "match analytic indices)")


def sobol_required_exposure(p: SpeciesParams, m: int = 1024, r: int = 1024) -> dict:
    """Sobol indices of D_required (geometric mean across tissues) w.r.t.
    t_half, P_liver, P_muscle, P_egg, k_ovo. ka drops out at steady state."""
    rng = np.random.default_rng(20260913)
    bounds = {
        "t_half_d": (0.5, 14.0),
        "P_liver": (0.5, 10.0),
        "P_muscle": (0.5, 8.0),
        "P_egg": (0.5, 8.0),
        "k_ovo": (1.0 / 30.0, 1.0 / 20.0),
    }

    def f(S):
        th, pl, pm, pe, ko = (S[:, i] for i in range(S.shape[1]))
        kt = np.log(2.0) / (th * 24.0)
        ed = p.Q["egg"] + ko * p.V["egg"] * pe
        es = p.Q["egg"] * ko * p.V["egg"] * pe / ed
        cp = (1.0 / 24.0) / (kt * p.V["plasma"] + es)
        tgt = TARGETS[p.name]
        d = np.stack([tgt["liver"] / (pl * (cp + (1.0 / 24.0) / p.Q["liver"])),
                      tgt["muscle"] / (pm * cp),
                      tgt["egg"] / (pe * p.Q["egg"] * cp / ed)])
        return np.exp(np.mean(np.log(d), axis=0))

    S1, ST = sobol_indices_direct(f, bounds, m=m, r=r, rng=rng)
    return {"names": list(bounds), "S1": S1.tolist(), "ST": ST.tolist()}


# --- Reporting -------------------------------------------------------------------

def analyse_species(p: SpeciesParams) -> dict:
    print(f"\n=== {p.name.upper()} ===")
    ss_err = check_steady_state_vs_integration(p)
    print(f"analytic steady state vs ODE integration: max rel. dev = {ss_err:.2e}")
    assert ss_err < 1e-4, "analytic steady state disagrees with integration"

    unit = unit_predictions(p, T_HALF_DEFAULT_DAYS)
    d_req = required_exposure(p, T_HALF_DEFAULT_DAYS)
    pf = pattern_consistency(p, T_HALF_DEFAULT_DAYS)
    hus = HUSBANDRY[p.name]
    d_hat = float(np.exp(np.mean(np.log(list(d_req.values())))))

    print(f"\nunit-dose steady state (1 ug/kg BW/d, t_half={T_HALF_DEFAULT_DAYS:g} d):")
    for t in ("plasma", "liver", "kidney", "muscle", "egg"):
        print(f"  {t:>7s}: {unit[t]:8.2f} ug/L")
    print(f"\nrequired exposure (ug/kg BW/d) per tissue: "
          + ", ".join(f"{t}={v:.4f}" for t, v in d_req.items()))
    print(f"pattern consistency factor (max/min across tissues): {pf:.2f}")
    print(f"geometric-mean required exposure: {d_hat:.4f} ug/kg BW/d")
    print(f"  -> feed  ~{d_hat * hus['BW_kg'] / hus['feed_kg_day'] * 1000:.0f} ng/kg feed"
          f"   water ~{d_hat * hus['BW_kg'] / hus['water_L_day'] * 1000:.0f} ng/L")

    mc = mc_required_exposure(p)
    d_comb = np.exp(np.mean(np.log(np.stack([mc["liver"], mc["muscle"], mc["egg"]])), axis=0))
    lo, med, hi = np.percentile(d_comb, [2.5, 50, 97.5])
    print(f"MC combined required exposure: median {med:.4f}, 95% CI [{lo:.4f}, {hi:.4f}] ug/kg BW/d")
    print(f"  -> feed median {med * hus['BW_kg'] / hus['feed_kg_day'] * 1000:.0f} ng/kg "
          f"(95% CI [{lo * hus['BW_kg'] / hus['feed_kg_day'] * 1000:.0f}, "
          f"{hi * hus['BW_kg'] / hus['feed_kg_day'] * 1000:.0f}])")

    sweep = {float(th): required_exposure(p, th) for th in T_HALF_SWEEP}
    print("\nrequired exposure vs assumed t_half (ug/kg BW/d, geometric mean):")
    for th, d in sweep.items():
        print(f"  t_half={th:5.1f} d: {np.exp(np.mean(np.log(list(d.values())))):.4f}")

    fsens = formulation_sensitivity(p)
    print(f"egg-formulation sensitivity: baseline d_hat={fsens['baseline']['d_hat']:.3g}, "
          f"no-sink d_hat={fsens['no_sink']['d_hat']:.3g} "
          f"(ratio {fsens['ratio_no_sink_over_baseline']:.1f}x); pattern factors "
          f"{fsens['baseline']['pattern_factor']:.2f} / {fsens['no_sink']['pattern_factor']:.2f}")
    asens = absorption_sensitivity(p)
    print("absorption sensitivity (F -> d_hat): "
          + ", ".join(f"F={k:g}:{v:.3g}" for k, v in asens.items()))

    return {"steady_state_check": ss_err, "unit": unit, "d_req": d_req,
            "pattern_factor": pf, "d_hat": d_hat,
            "mc": {"median": med, "lo": lo, "hi": hi},
            "sweep": sweep, "sobol": sobol_required_exposure(p),
            "absorption_sensitivity": absorption_sensitivity(p),
            "formulation_sensitivity": formulation_sensitivity(p)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sobol_self_test()
    results = {name: analyse_species(p) for name, p in
               (("chicken", chicken_params()), ("duck", duck_params()))}

    # ---- Figures -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) predicted vs observed pattern at the geometric-mean exposure
    ax = axes[0]
    tissues = ["liver", "muscle", "egg"]
    width = 0.35
    for j, (name, res) in enumerate(results.items()):
        scaled = [res["unit"][t] * res["d_hat"] for t in tissues]
        obs = [TARGETS[name][t] for t in tissues]
        x = np.arange(len(tissues)) + (j - 0.5) * width
        ax.bar(x - width / 2, scaled, width, label=f"{name} model", alpha=0.9)
        ax.bar(x + width / 2, obs, width, label=f"{name} observed", alpha=0.9,
               hatch="//", edgecolor="white")
    ax.set_xticks(np.arange(len(tissues)), tissues)
    ax.set_ylabel("Concentration (ug/kg wet weight)")
    ax.set_yscale("log")
    ax.set_title("(a) Steady-state pattern vs survey means\n(at the back-calculated exposure)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (b) required exposure vs assumed half-life
    ax = axes[1]
    for name, res in results.items():
        ths = sorted(res["sweep"])
        d = [np.exp(np.mean(np.log(list(res["sweep"][str(th)].values())))) if str(th) in res["sweep"]
             else np.exp(np.mean(np.log(list(res["sweep"][th].values())))) for th in ths]
        ax.plot(ths, d, "o-", label=name)
    ax.set_xlabel("Assumed elimination half-life (d)")
    ax.set_ylabel("Required exposure (ug/kg BW/d)")
    ax.set_title("(b) Required exposure vs assumed half-life\n(weak dependence: oviposition dominates clearance)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Sobol total-order indices for chicken D_hat
    ax = axes[2]
    sb = results["chicken"]["sobol"]
    x = np.arange(len(sb["names"]))
    ax.bar(x - 0.18, sb["S1"], 0.35, label="S1 (first-order)")
    ax.bar(x + 0.18, sb["ST"], 0.35, label="ST (total-order)")
    ax.set_xticks(x, sb["names"], rotation=30, ha="right")
    ax.set_ylabel("Sobol index")
    ax.set_title("(c) Sensitivity of required exposure\n(chicken; duck similar)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Reverse dosimetry with the rebuilt poultry PFOA PBPK model\n"
                 "(assumed partition coefficients and husbandry; conditional exposure estimate)",
                 fontsize=11)
    fig.tight_layout()
    fig_path = OUT / "reverse_dosimetry_summary.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\nFigure written to {fig_path}")

    # ---- JSON + CSV ----------------------------------------------------------
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    json_path = OUT / "reverse_dosimetry_results.json"
    json_path.write_text(json.dumps(clean(results), indent=2), encoding="utf-8")
    print(f"JSON written to {json_path}")

    csv_path = OUT / "reverse_dosimetry_required_exposure.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("species,tissue,required_exposure_ug_per_kgBW_day,t_half_assumed_d\n")
        for name, res in results.items():
            for t, v in res["d_req"].items():
                fh.write(f"{name},{t},{v:.6g},{T_HALF_DEFAULT_DAYS}\n")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
