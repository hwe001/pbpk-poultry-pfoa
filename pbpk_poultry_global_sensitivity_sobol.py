import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define the PBPK model equations
def pbpk_model(t, y, params):
    C_gut, C_plasma, C_liver, C_kidney, C_muscle, C_fat, C_egg, C_rest = y

    Ka = params["Ka"]
    Q_liver = params["Q_liver"]
    P_liver = params["P_liver"]
    K_met = params["K_met"]
    Q_kidney = params["Q_kidney"]
    P_kidney = params["P_kidney"]
    K_urine = params["K_urine"]
    Q_muscle = params["Q_muscle"]
    P_muscle = params["P_muscle"]
    Q_fat = params["Q_fat"]
    P_fat = params["P_fat"]
    Q_egg = params["Q_egg"]
    P_egg = params["P_egg"]
    Q_rest = params["Q_rest"]
    P_rest = params["P_rest"]

    dC_gut = -Ka * C_gut
    # NOTE: matches pbpk_poultry_revised.py's liver equation exactly (dosed gut
    # feeds liver directly via Ka*C_gut, not just via plasma clearance)
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

# Parameters matching the validated model in pbpk_poultry_revised.py exactly
# (the earlier version of this script used a stale, differently-calibrated
# parameter set - absolute L/h flow values from an older parameter table,
# not the fractional-of-cardiac-output values that actually produced the
# paper's validated Figure 1 results)
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

# Time points for simulation (700 hours)
t_eval_full = np.linspace(0, 700, 1400)

# Multi-dose schedule matching pbpk_poultry_revised.py: 7 doses of 0.0005 mg
# (0.5 ug) every 24 hours up to 144 hours - the earlier version of this
# script used dose_amount=0.5 (mg), a 1000x scale mismatch against the
# validated model.
dose_times = np.arange(0, 145, 24)  # [0, 24, 48, ..., 144]
dose_amount = 0.0005
n_doses = len(dose_times)

def run_simulation_with_doses(params):
    y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    t_all = []
    y_all = []

    for i in range(n_doses):
        t_start = dose_times[i]
        t_end = dose_times[i + 1] if i + 1 < n_doses else 700
        t_eval = np.linspace(t_start, t_end, int((t_end - t_start) / (700 / len(t_eval_full))))

        y0[0] += dose_amount

        sol = solve_ivp(lambda t, y: pbpk_model(t, y, params), [t_start, t_end], y0, t_eval=t_eval, method='Radau')

        if not sol.success:
            print(f"Solver failed for parameters: K_urine={params['K_urine']}, P_liver={params['P_liver']}, P_muscle={params['P_muscle']}, P_egg={params['P_egg']}")
            return np.array(t_all), np.full((8, len(t_all)), np.nan)

        t_all.extend(sol.t)
        y_all.append(sol.y)

        y0 = sol.y[:, -1].copy()

    y_all = np.hstack(y_all)
    return np.array(t_all), y_all

# Sobol problem definition: 4 parameters, +-30% ranges (within the
# manuscript's stated "ranges of +-20-40%" - previously this script used a
# mix of +-20% for K_urine and +-15% for the partition coefficients, neither
# matching the stated range; +-30% also matches the Monte Carlo perturbation
# used elsewhere in the manuscript for K_urine/P_liver, for consistency)
RANGE_FRAC = 0.30

def pm_bounds(nominal):
    return [nominal * (1 - RANGE_FRAC), nominal * (1 + RANGE_FRAC)]

problem_chicken = {
    'num_vars': 4,
    'names': ['K_urine', 'P_liver', 'P_muscle', 'P_egg'],
    'bounds': [
        pm_bounds(params_chicken["K_urine"]),
        pm_bounds(params_chicken["P_liver"]),
        pm_bounds(params_chicken["P_muscle"]),
        pm_bounds(params_chicken["P_egg"]),
    ]
}

problem_duck = {
    'num_vars': 4,
    'names': ['K_urine', 'P_liver', 'P_muscle', 'P_egg'],
    'bounds': [
        pm_bounds(params_duck["K_urine"]),
        pm_bounds(params_duck["P_liver"]),
        pm_bounds(params_duck["P_muscle"]),
        pm_bounds(params_duck["P_egg"]),
    ]
}

# Saltelli sampling matching the manuscript's stated "1024 samples ...
# 6144 simulations": N*(D+2) = 1024*(4+2) = 6144 requires
# calc_second_order=False (the previous script used N=100 or N=1000 with
# calc_second_order defaulting True, giving 1000-10000 samples, matching
# neither the stated sample count nor simulation count)
N_SALTELLI = 1024
param_values_chicken = saltelli.sample(problem_chicken, N_SALTELLI, calc_second_order=False)
param_values_duck = saltelli.sample(problem_duck, N_SALTELLI, calc_second_order=False)
print(f"Chicken: {param_values_chicken.shape[0]} total simulations (target: 6144)")
print(f"Duck: {param_values_duck.shape[0]} total simulations (target: 6144)")

def evaluate_model(params_base, param_values):
    results = {"Liver": [], "Muscle": [], "Egg": []}

    for i in range(param_values.shape[0]):
        params = params_base.copy()
        params["K_urine"] = param_values[i, 0]
        params["P_liver"] = param_values[i, 1]
        params["P_muscle"] = param_values[i, 2]
        params["P_egg"] = param_values[i, 3]

        t, y = run_simulation_with_doses(params)

        liver_conc = y[2, -1]
        muscle_conc = y[4, -1]
        egg_conc = y[6, -1]

        if np.isnan(liver_conc) or np.isnan(muscle_conc) or np.isnan(egg_conc):
            print(f"NaN detected for parameters: K_urine={params['K_urine']}, P_liver={params['P_liver']}, P_muscle={params['P_muscle']}, P_egg={params['P_egg']}")
            liver_conc = 0.0 if np.isnan(liver_conc) else liver_conc
            muscle_conc = 0.0 if np.isnan(muscle_conc) else muscle_conc
            egg_conc = 0.0 if np.isnan(egg_conc) else egg_conc

        results["Liver"].append(liver_conc)
        results["Muscle"].append(muscle_conc)
        results["Egg"].append(egg_conc)

    return results

# num_resamples controls SALib's internal bootstrap for S1_conf/ST_conf (95%
# CI half-widths) - this is the real per-parameter uncertainty estimate that
# the existing Figure 3 caption already claims to show, but neither the
# original nor the first-pass corrected script actually computed it (the
# error bars previously visible were a seaborn artifact from auto-bootstrapping
# over just the 2 species values per bar, not a real Sobol CI)
NUM_RESAMPLES = 1000

print("Running Chicken Sobol simulations...")
results_chicken = evaluate_model(params_chicken, param_values_chicken)

sobol_indices_chicken = {}
for tissue in ["Liver", "Muscle", "Egg"]:
    Si = sobol.analyze(problem_chicken, np.array(results_chicken[tissue]), calc_second_order=False,
                        num_resamples=NUM_RESAMPLES, print_to_console=False)
    sobol_indices_chicken[tissue] = {
        "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
        "Parameter": problem_chicken["names"]
    }

print("Running Duck Sobol simulations...")
results_duck = evaluate_model(params_duck, param_values_duck)

sobol_indices_duck = {}
for tissue in ["Liver", "Muscle", "Egg"]:
    Si = sobol.analyze(problem_duck, np.array(results_duck[tissue]), calc_second_order=False,
                        num_resamples=NUM_RESAMPLES, print_to_console=False)
    sobol_indices_duck[tissue] = {
        "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
        "Parameter": problem_duck["names"]
    }

# Prepare data for plotting and CSV export
sobol_data_s1 = []
sobol_data_st = []
for species, indices in [("Chicken", sobol_indices_chicken), ("Duck", sobol_indices_duck)]:
    for tissue in ["Liver", "Muscle", "Egg"]:
        for i, param in enumerate(indices[tissue]["Parameter"]):
            sobol_data_s1.append({
                "Species": species, "Tissue": tissue, "Parameter": param,
                "First-Order Index (S1)": indices[tissue]["S1"][i],
                "S1_conf": indices[tissue]["S1_conf"][i]
            })
            sobol_data_st.append({
                "Species": species, "Tissue": tissue, "Parameter": param,
                "Total-Order Index (ST)": indices[tissue]["ST"][i],
                "ST_conf": indices[tissue]["ST_conf"][i]
            })

sobol_df_s1 = pd.DataFrame(sobol_data_s1)
sobol_df_st = pd.DataFrame(sobol_data_st)

merged = sobol_df_s1.merge(sobol_df_st, on=["Species", "Tissue", "Parameter"])
merged.to_csv("sobol_results_corrected.csv", index=False)
print("Saved sobol_results_corrected.csv")

# Plot with REAL bootstrap 95% CI error bars (SALib's S1_conf/ST_conf),
# shown separately per species since a bootstrap CI is only meaningful
# within one species' own resampled simulation ensemble, not averaged
# across the two species' point estimates (which is what the previous
# version of this figure effectively did via seaborn's automatic CI).
plt.style.use('grayscale')
tissues = ["Liver", "Muscle", "Egg"]
params = problem_chicken["names"]
n_param = len(params)
bar_h = 0.25
y_base = np.arange(n_param)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
species_list = ["Chicken", "Duck"]
metrics = [("First-Order Index (S1)", "S1_conf", sobol_df_s1), ("Total-Order Index (ST)", "ST_conf", sobol_df_st)]

for col, species in enumerate(species_list):
    for row, (metric_col, conf_col, df) in enumerate(metrics):
        ax = axes[row, col]
        sub = df[df["Species"] == species]
        for ti, tissue in enumerate(tissues):
            tsub = sub[sub["Tissue"] == tissue].set_index("Parameter").reindex(params)
            y_pos = y_base + (ti - 1) * bar_h
            ax.barh(y_pos, tsub[metric_col], height=bar_h, xerr=tsub[conf_col],
                    label=tissue, capsize=2)
        ax.set_yticks(y_base)
        ax.set_yticklabels(params)
        ax.set_xlabel(metric_col, fontsize=11)
        ax.set_title(f"{species}", fontsize=12)
        if col == 0:
            ax.set_ylabel("Parameter", fontsize=11)
        if row == 0 and col == 1:
            ax.legend(title="Tissue", fontsize=9, title_fontsize=10)

fig.suptitle("Sobol Sensitivity Indices (corrected parameters/sample size, real bootstrap 95% CI)", fontsize=13)
plt.tight_layout()
plt.savefig("Sobol_Indices_corrected.png", dpi=300, bbox_inches='tight')
print("Saved Sobol_Indices_corrected.png")

print("\nSobol Sensitivity Indices (Chicken):")
for tissue in sobol_indices_chicken:
    print(f"\nTissue: {tissue}")
    print("First-Order Indices (S1):")
    for param, s1 in zip(sobol_indices_chicken[tissue]["Parameter"], sobol_indices_chicken[tissue]["S1"]):
        print(f"{param}: {s1:.3f}")
    print("Total-Order Indices (ST):")
    for param, st in zip(sobol_indices_chicken[tissue]["Parameter"], sobol_indices_chicken[tissue]["ST"]):
        print(f"{param}: {st:.3f}")

print("\nSobol Sensitivity Indices (Duck):")
for tissue in sobol_indices_duck:
    print(f"\nTissue: {tissue}")
    print("First-Order Indices (S1):")
    for param, s1 in zip(sobol_indices_duck[tissue]["Parameter"], sobol_indices_duck[tissue]["S1"]):
        print(f"{param}: {s1:.3f}")
    print("Total-Order Indices (ST):")
    for param, st in zip(sobol_indices_duck[tissue]["Parameter"], sobol_indices_duck[tissue]["ST"]):
        print(f"{param}: {st:.3f}")
