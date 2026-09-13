# -*- coding: utf-8 -*-
"""Local sensitivity analysis for the validated PBPK poultry PFOA model
(matching pbpk_poultry_revised.py's parameters exactly). Each parameter is
perturbed +10% from its nominal value, holding all others fixed, and the
sensitivity index S = (%change in steady-state tissue concentration) /
(%change in parameter) is computed per Methods (a +-10% perturbation and
Eq. S = dC/C / dP/P). This replaces the previous version of this script,
which only plotted a hardcoded results table with no underlying
computation to verify or reproduce it.
"""
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

def pbpk_model(t, y, params):
    C_gut, C_plasma, C_liver, C_kidney, C_muscle, C_fat, C_egg, C_rest = y
    Ka = params["Ka"]; Q_liver = params["Q_liver"]; P_liver = params["P_liver"]
    K_met = params["K_met"]; Q_kidney = params["Q_kidney"]; P_kidney = params["P_kidney"]
    K_urine = params["K_urine"]; Q_muscle = params["Q_muscle"]; P_muscle = params["P_muscle"]
    Q_fat = params["Q_fat"]; P_fat = params["P_fat"]; Q_egg = params["Q_egg"]
    P_egg = params["P_egg"]; Q_rest = params["Q_rest"]; P_rest = params["P_rest"]

    dC_gut = -Ka * C_gut
    dC_liver = Ka * C_gut + Q_liver * (C_plasma - C_liver / P_liver) - K_met * C_liver
    dC_plasma = (
        Q_liver * (C_liver / P_liver - C_plasma)
        + Q_kidney * (C_kidney / P_kidney - C_plasma)
        + Q_muscle * (C_muscle / P_muscle - C_plasma)
        + Q_fat * (C_fat / P_fat - C_plasma)
        + Q_egg * (C_egg / P_egg - C_plasma)
        + Q_rest * (C_rest / P_rest - C_plasma)
        - K_urine * C_plasma
    )
    dC_kidney = Q_kidney * (C_plasma - C_kidney / P_kidney) - K_urine * C_kidney
    dC_muscle = Q_muscle * (C_plasma - C_muscle / P_muscle)
    dC_fat = Q_fat * (C_plasma - C_fat / P_fat)
    dC_egg = Q_egg * (C_plasma - C_egg / P_egg)
    dC_rest = Q_rest * (C_plasma - C_rest / P_rest)
    return [dC_gut, dC_plasma, dC_liver, dC_kidney, dC_muscle, dC_fat, dC_egg, dC_rest]

params_chicken = {
    "Ka": 0.5, "Q_liver": 0.25, "P_liver": 2.5, "K_met": 0.0005, "Q_kidney": 0.143,
    "P_kidney": 0.6, "K_urine": 0.015, "Q_muscle": 0.214, "P_muscle": 2.0,
    "Q_fat": 0.071, "P_fat": 5.0, "Q_egg": 0.036, "P_egg": 1.5, "Q_rest": 0.286, "P_rest": 1.2
}
params_duck = {
    "Ka": 0.4, "Q_liver": 0.25, "P_liver": 2.2, "K_met": 0.0006, "Q_kidney": 0.150,
    "P_kidney": 0.55, "K_urine": 0.0125, "Q_muscle": 0.208, "P_muscle": 2.0,
    "Q_fat": 0.067, "P_fat": 4.5, "Q_egg": 0.033, "P_egg": 1.5, "Q_rest": 0.292, "P_rest": 1.1
}

dose_times = np.arange(0, 145, 24)
dose_amount = 0.0005
n_doses = len(dose_times)
t_eval_full = np.linspace(0, 700, 1400)

def run_simulation_with_doses(params):
    y0 = np.array([0.0]*8)
    for i in range(n_doses):
        t_start = dose_times[i]
        t_end = dose_times[i + 1] if i + 1 < n_doses else 700
        t_eval = np.linspace(t_start, t_end, int((t_end - t_start) / (700 / len(t_eval_full))))
        y0[0] += dose_amount
        sol = solve_ivp(lambda t, y: pbpk_model(t, y, params), [t_start, t_end], y0, t_eval=t_eval, method='Radau')
        y0 = sol.y[:, -1].copy()
    return y0  # final steady-state compartment amounts/concentrations

TISSUE_IDX = {"Liver": 2, "Muscle": 4, "Egg": 6}
PERTURB_FRAC = 0.10
PARAMS_TO_TEST = ["Ka", "K_urine", "K_met", "P_liver", "P_muscle", "P_egg"]

def local_sensitivity(params_base, species):
    baseline = run_simulation_with_doses(params_base)
    rows = []
    for param in PARAMS_TO_TEST:
        params_pert = params_base.copy()
        params_pert[param] = params_base[param] * (1 + PERTURB_FRAC)
        perturbed = run_simulation_with_doses(params_pert)
        for tissue, idx in TISSUE_IDX.items():
            c0, c1 = baseline[idx], perturbed[idx]
            pct_change_c = (c1 - c0) / c0
            S = pct_change_c / PERTURB_FRAC
            rows.append({"Species": species, "Parameter": param, "Tissue": tissue, "Sensitivity": S})
    return rows

print("Running chicken local sensitivity...")
rows = local_sensitivity(params_chicken, "Chicken")
print("Running duck local sensitivity...")
rows += local_sensitivity(params_duck, "Duck")

df = pd.DataFrame(rows)
df.to_csv("local_sensitivity_results_corrected.csv", index=False)
print("Saved local_sensitivity_results_corrected.csv")
print(df.to_string(index=False))
