#!/usr/bin/env python3
"""Rebuilt flow-limited PBPK model of PFOA in poultry (chicken / duck).

WHY A REBUILD
-------------
Review of the original scripts (github.com/hwe001/pbpk-poultry-pfoa) and the
manuscript's printed equations found structural defects that a parameter
refit cannot repair:

  1. Renal clearance was applied twice (from the plasma equation AND from the
     kidney equation) -- one clearance process, two sinks.
  2. "Flows" were fractions of cardiac output (summing to 1, dimensionless)
     used as if they were flow rates; no cardiac output, no tissue volumes
     anywhere, so the equations could not conserve mass.
  3. The dosing code added an amount (mg) directly into a concentration
     state -- dimensionally undefined.
  4. A fitted hepatic metabolism rate was included although PFOA is not
     metabolised (a free fudge parameter).

This module implements the standard well-stirred, flow-limited PBPK form
instead. State variables are AMOUNTS; concentrations are derived amounts /
volumes:

    gut lumen:   dA_gut/dt     = input(t) - ka * A_gut
    tissue i:    dA_i/dt       = Q_i * (C_plasma - C_i / P_i)
    liver:       dA_liv/dt     = Q_liv * (C_plasma - C_liv / P_liv) + ka * A_gut
    egg:         dA_egg/dt     = Q_egg * (C_plasma - C_egg / P_egg) - k_ovo * A_egg
    plasma:      dA_plasma/dt  = sum_i Q_i * (C_i / P_i - C_plasma) - k_u * A_plasma

with C_i = A_i / V_i, plus two bookkeeping states accumulating renally
excreted and oviposited amounts. The system is linear, so a mass-balance
identity (dose in = stored + eliminated) holds exactly up to solver
tolerance; `mass_balance_error()` quantifies it for every run and
`test_mass_balance.py` asserts it.

UNITS (body-weight normalised, BW = 1 kg)
-----------------------------------------
  amounts        ug/kg BW
  volumes        L/kg BW
  blood flows    L/h/kg BW   (QCO * fraction)
  concentrations ug/L
  dose           ug/kg BW per bolus

For a bird of body weight W kg, multiply dose amounts by W; concentrations
are unchanged (linearity).

ELIMINATION
-----------
PFOA is not metabolised: there is no metabolism term (the legacy fitted
K_met is dropped, not retuned). Renal excretion is a single first-order loss
from the plasma pool, rate k_u * A_plasma. Physiologically this lumps
filtration, secretion and net reabsorption (PFOA is heavily reabsorbed via
OAT transporters, hence its long half-life); k_u is therefore a NET rate,
not a filtration clearance.

EGG COMPARTMENT
---------------
A laying bird's market egg reflects ~1 day of deposition followed by
oviposition. The egg is modelled as a perfused compartment (follicle +
oviduct lumped) with an additional first-order loss k_ovo representing
oviposition; laid egg mass is accumulated in a tracking state. This is a
deliberate simplification of sequential follicular development -- adequate
for quasi-steady egg concentrations under ongoing exposure, not for
resolving the content of individual eggs.

ASSUMPTION FLAGS
----------------
Every parameter that is not a documented literature value for the species is
flagged `# ASSUMPTION` with its rationale, as in this group's other models.
Volumes and cardiac output in particular need verification against
Wang et al. (2021) Part II, which the legacy manuscript cites for flows and
organ weights but from which we do not currently have transcribed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp

# --- State layout -----------------------------------------------------------
# 0 A_gut, 1 A_plasma, 2 A_liver, 3 A_kidney, 4 A_muscle, 5 A_fat, 6 A_egg,
# 7 A_rest, 8 cum_renal (bookkeeping), 9 cum_egg_out (bookkeeping)
N_STATES = 10
I_GUT, I_PLASMA, I_LIVER, I_KIDNEY, I_MUSCLE, I_FAT, I_EGG, I_REST = range(8)

COMPARTMENTS = ("gut", "plasma", "liver", "kidney", "muscle", "fat", "egg", "rest")

# --- Physiological parameters (per kg BW) -----------------------------------
# Blood-flow fractions of cardiac output, as normalised in the legacy Table 1
# (attributed to Wang et al. 2021, normalised). Gut absorption is portal mass
# input, so no separate GI-tissue flow is used.
FLOW_FRACTIONS = {
    "liver": 0.25, "kidney": 0.143, "muscle": 0.214,
    "fat": 0.071, "egg": 0.036, "rest": 0.286,
}  # sums to 1.000 by construction

Q_CO = 10.0  # L/h/kg BW. ASSUMPTION: avian cardiac output ~167 mL/min/kg.
             # Verify against Wang et al. (2021) Part II before publication.

VOLUMES = {
    "plasma": 0.045,  # 4.5% BW. ASSUMPTION (avian plasma volume ~4-5% BW).
    "liver": 0.0214,  # 2.14% BW, quoted from Wang et al. (2021) in the manuscript.
    "kidney": 0.006,  # 0.6% BW. ASSUMPTION (typical avian kidney).
    "muscle": 0.40,   # 40% BW. ASSUMPTION (laying hen total skeletal muscle).
    "fat": 0.07,      # 7% BW. ASSUMPTION (laying hen adipose).
    "egg": 0.05,      # one forming egg ~50 g. ASSUMPTION.
}
VOLUMES["rest"] = 1.0 - sum(VOLUMES.values())  # remainder of a 1-kg body

# --- Species parameter sets ---------------------------------------------------
# ka, k_u and the partition coefficients are CALIBRATION parameters (initial
# values below are the legacy table's, carried over as starting guesses only;
# they will be refit after the calibration-target units are confirmed).
# k_met from the legacy table is intentionally absent: PFOA is not metabolised.

@dataclass
class SpeciesParams:
    name: str
    ka: float                 # 1/h, first-order gut absorption
    k_u: float                # 1/h, net renal elimination from plasma pool
    k_ovo: float              # 1/h, oviposition rate
    P: dict[str, float]       # tissue:plasma partition coefficients
    Q: dict[str, float] = field(default_factory=lambda: {k: v * Q_CO
                                                         for k, v in FLOW_FRACTIONS.items()})
    V: dict[str, float] = field(default_factory=lambda: dict(VOLUMES))

    def with_overrides(self, **overrides) -> "SpeciesParams":
        return replace(self, **overrides)


def chicken_params() -> SpeciesParams:
    return SpeciesParams(
        name="chicken",
        ka=0.5,          # legacy fitted value, starting guess only
        k_u=0.015,       # legacy fitted value, starting guess only
        k_ovo=1.0 / 25.0,  # ASSUMPTION: ~1 egg / 25 h in lay
        P={"liver": 2.5, "kidney": 0.6, "muscle": 2.0,
           "fat": 5.0, "egg": 1.5, "rest": 1.2},  # legacy guesses
    )


def duck_params() -> SpeciesParams:
    return SpeciesParams(
        name="duck",
        ka=0.4,          # legacy fitted value, starting guess only
        k_u=0.0125,      # legacy fitted value, starting guess only
        k_ovo=1.0 / 26.0,  # ASSUMPTION: ~1 egg / 26 h in lay
        P={"liver": 2.2, "kidney": 0.55, "muscle": 2.0,
           "fat": 4.5, "egg": 1.5, "rest": 1.1},  # legacy guesses
    )


# --- Model core --------------------------------------------------------------

def rhs(t: float, y: np.ndarray, p: SpeciesParams) -> np.ndarray:
    A = y[:8]
    C = {name: A[i] / p.V[name] for name, i in
         zip(COMPARTMENTS[1:], range(1, 8))}  # plasma..rest (gut is a lumen, no volume)

    cp = C["plasma"]
    to_plasma = 0.0
    for name in ("liver", "kidney", "muscle", "fat", "egg", "rest"):
        to_plasma += p.Q[name] * (C[name] / p.P[name] - cp)

    d = np.zeros(N_STATES)
    d[I_GUT] = -p.ka * A[I_GUT]
    d[I_PLASMA] = to_plasma - p.k_u * A[I_PLASMA]
    for name, idx in (("liver", I_LIVER), ("kidney", I_KIDNEY),
                      ("muscle", I_MUSCLE), ("fat", I_FAT), ("rest", I_REST)):
        d[idx] = p.Q[name] * (cp - C[name] / p.P[name])
    d[I_LIVER] += p.ka * A[I_GUT]  # portal input: absorbed dose enters the liver
    d[I_EGG] = p.Q["egg"] * (cp - C["egg"] / p.P["egg"]) - p.k_ovo * A[I_EGG]
    d[8] = p.k_u * A[I_PLASMA]          # cumulative renal excretion
    d[9] = p.k_ovo * A[I_EGG]           # cumulative oviposited amount
    return d


@dataclass
class Solution:
    species: str
    t: np.ndarray            # h
    amounts: np.ndarray      # (n_states, n_times), ug/kg BW
    dose_total: float        # ug/kg BW administered
    mass_err: float          # fractional mass-balance error (see below)

    def conc(self, compartment: str) -> np.ndarray:
        """Concentration in ug/L. 'gut' returns the lumen amount (no volume)."""
        if compartment == "gut":
            return self.amounts[I_GUT]
        return self.amounts[I_PLASMA + COMPARTMENTS.index(compartment) - 1] / VOLUMES[compartment]

    def cum_renal(self) -> np.ndarray:
        return self.amounts[8]

    def cum_egg_out(self) -> np.ndarray:
        return self.amounts[9]


def simulate(params: SpeciesParams, dose_times, dose_amounts, t_end: float,
             n_points: int = 1401) -> Solution:
    """Integrate, resetting A_gut upward at each oral bolus (piecewise segments).

    dose_times/dose_amounts: bolus schedule (ug/kg BW). Segments between boluses
    are integrated separately with tight tolerances (Radau), as the legacy
    code did, but with mass balance checked across the whole run.
    """
    dose_times = np.atleast_1d(np.asarray(dose_times, dtype=float))
    dose_amounts = np.atleast_1d(np.asarray(dose_amounts, dtype=float))
    if dose_times.shape != dose_amounts.shape:
        raise ValueError("dose_times and dose_amounts must have equal length")
    if dose_times.size == 0:  # zero-dose run: single pre-dose segment
        dose_times = np.array([0.0])
        dose_amounts = np.array([0.0])
    elif dose_times[0] > 0:
        dose_times = np.concatenate(([0.0], dose_times))
        dose_amounts = np.concatenate(([0.0], dose_amounts))  # first segment pre-dose

    y = np.zeros(N_STATES)
    dose_total = float(dose_amounts.sum())
    t_all, y_all = [], []
    boundaries = np.concatenate((dose_times, [t_end]))
    for k in range(len(boundaries) - 1):
        t0, t1 = boundaries[k], boundaries[k + 1]
        if t1 <= t0:
            continue
        y[I_GUT] += dose_amounts[k]  # bolus into gut lumen at segment start
        n_seg = max(2, int(round((t1 - t0) * (n_points - 1) / t_end)) + 1)
        t_eval = np.linspace(t0, t1, n_seg)
        sol = solve_ivp(rhs, (t0, t1), y, args=(params,), method="Radau",
                        t_eval=t_eval, rtol=1e-8, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f"Integration failed on [{t0}, {t1}]: {sol.message}")
        t_all.append(sol.t)
        y_all.append(sol.y)
        y = sol.y[:, -1].copy()

    t = np.concatenate(t_all)
    amounts = np.hstack(y_all)
    stored = amounts[:8, -1].sum()
    eliminated = amounts[8, -1] + amounts[9, -1]
    mass_err = abs(dose_total - stored - eliminated) / max(dose_total, 1e-300)
    return Solution(params.name, t, amounts, dose_total, mass_err)


def legacy_scenario(params: SpeciesParams, dose_per_bolus: float = 0.5,
                    n_boluses: int = 7, t_end: float = 700.0) -> Solution:
    """The legacy calibration scenario: daily oral boluses for 7 days.

    NOTE the legacy code injected 0.0005 *mg* into a concentration state with
    no defined volume; here the same nominal regimen is expressed properly as
    0.5 ug/kg BW per bolus (equivalent assuming a 1-kg BW-normalised bird).
    Absolute-dose equivalence with the legacy runs is NOT claimed -- the legacy
    dose units were undefined.
    """
    return simulate(params, dose_times=np.arange(0, 24 * n_boluses, 24.0),
                    dose_amounts=np.full(n_boluses, dose_per_bolus), t_end=t_end)
