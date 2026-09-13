#This code is the supplementary code for the paper:
# Ho, Tang, Bai, Zhang "Physiologically Based Pharmacokinetic Modeling of PFOA
# in Poultry: Insights into Tissue Distribution"
# please contact corresponding author Dr Harvey Ho: harvey.ho@auckland.ac.nz
# if you would like to use the code.
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Check Matplotlib version for debugging
print(f"Matplotlib version: {matplotlib.__version__}")

# Define the updated PBPK model equations
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
    
    # Gut: Absorbed compound goes to the plasma
    dC_gut = -Ka * C_gut
    
    # Liver: Receives compound from the plasma only
    dC_liver = Q_liver * (C_plasma - C_liver / P_liver) - K_met * C_liver
    
    # Plasma: Receives absorbed compound from the gut
    dC_plasma = (
        Ka * C_gut
        + Q_liver * (C_liver / P_liver - C_plasma)
        + Q_kidney * (C_kidney / P_kidney - C_plasma)
        + Q_muscle * (C_muscle / P_muscle - C_plasma)
        + Q_fat * (C_fat / P_fat - C_plasma)
        + Q_egg * (C_egg / P_egg - C_plasma)
        + Q_rest * (C_rest / P_rest - C_plasma)
        - K_urine * C_plasma
    )
    
    # Other compartments
    dC_kidney = Q_kidney * (C_plasma - C_kidney / P_kidney) - K_urine * C_kidney
    dC_muscle = Q_muscle * (C_plasma - C_muscle / P_muscle)
    dC_fat = Q_fat * (C_plasma - C_fat / P_fat)
    dC_egg = Q_egg * (C_plasma - C_egg / P_egg)
    dC_rest = Q_rest * (C_plasma - C_rest / P_rest)
    
    return [dC_gut, dC_plasma, dC_liver, dC_kidney, dC_muscle, dC_fat, dC_egg, dC_rest]

# Updated parameters for chicken and duck with adjusted flow rates
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

# Time points for simulation (extended to 700 hours)
t_eval_full = np.linspace(0, 700, 1000)

# Multi-dose schedule: 7 doses of 0.0005 mg every 24 hours up to 144 hours
dose_times = np.arange(0, 145, 24)  # [0, 24, 48, ..., 144]
dose_amount = 0.0005
n_doses = len(dose_times)

# Function to run a single simulation with multi-dose
def run_simulation_with_doses(params):
    y0 = [0, 0, 0, 0, 0, 0, 0, 0]
    
    t_all = []
    y_all = []
    
    for i in range(n_doses):
        t_start = dose_times[i]
        t_end = dose_times[i + 1] if i + 1 < n_doses else 700
        t_eval = np.linspace(t_start, t_end, int((t_end - t_start) / (700 / len(t_eval_full))))
        
        y0[0] += dose_amount
        
        sol = solve_ivp(lambda t, y: pbpk_model(t, y, params), [t_start, t_end], y0, t_eval=t_eval)
        
        t_all.extend(sol.t)
        y_all.append(sol.y)
        
        y0 = sol.y[:, -1].copy()
    
    y_all = np.hstack(y_all)
    return np.array(t_all), y_all

# Monte Carlo simulation to estimate uncertainty
n_simulations = 500  # corrected to match manuscript's stated "500 simulations"
np.random.seed(42)

def run_monte_carlo(params, species):
    steady_states = []
    line_data = []
    for sim in range(n_simulations):
        params_var = params.copy()
        # Variability set to ±30%
        params_var["K_urine"] = np.random.normal(params["K_urine"], 0.3 * params["K_urine"])
        params_var["P_liver"] = np.random.normal(params["P_liver"], 0.3 * params["P_liver"])
        
        t, y = run_simulation_with_doses(params_var)
        
        C_steady = y[:, -1]
        steady_states.append({
            "Liver": C_steady[2],
            "Meat": C_steady[4],
            "Eggs": C_steady[6]
        })
        
        for tissue, idx, style in [("Liver", 2, "--"), ("Meat", 4, "--"), ("Eggs", 6, "--")]:
            if species == "Duck":
                style = "-."
            line_data.append({
                "Time (h)": t,
                "Concentration (mg/L)": y[idx],
                "Tissue": f"{species} {tissue}",
                "Style": style,
                "Simulation": sim
            })
    
    steady_data = []
    for tissue in ["Liver", "Meat", "Eggs"]:
        concentrations = [sim[tissue] for sim in steady_states]
        for conc in concentrations:
            steady_data.append({"Tissue": tissue, "Concentration (mg/L)": conc, "Species": species, "Source": "Model"})
    
    line_df = pd.DataFrame([
        {"Time (h)": t, "Concentration (mg/L)": c, "Tissue": tissue, "Style": style, "Simulation": sim}
        for d in line_data for t, c, tissue, style, sim in zip(d["Time (h)"], d["Concentration (mg/L)"], [d["Tissue"]] * len(d["Time (h)"]), [d["Style"]] * len(d["Time (h)"]), [d["Simulation"]] * len(d["Time (h)"]))
    ])
    
    return pd.DataFrame(steady_data), line_df

# Run Monte Carlo simulations for chicken and duck
chicken_steady, chicken_line = run_monte_carlo(params_chicken, "Chicken")
duck_steady, duck_line = run_monte_carlo(params_duck, "Duck")

# Combine line data for plotting
line_df = pd.concat([chicken_line, duck_line], ignore_index=True)

# Experimental data (corrected to match Chongqing study, in mg/L)
experimental_data = {
    "Chicken": {"Liver": 0.000251, "Meat": 0.000141, "Eggs": 0.000128},
    "Duck": {"Liver": 0.000173, "Meat": 0.000210, "Eggs": 0.000220},
}

# Create DataFrame for experimental data with synthetic variability (±5%)
experimental_df = []
for species, data in experimental_data.items():
    for tissue, mean_conc in data.items():
        concentrations = np.random.normal(mean_conc, 0.05 * mean_conc, n_simulations)
        for conc in concentrations:
            experimental_df.append({
                "Tissue": tissue,
                "Concentration (mg/L)": conc,
                "Species": species,
                "Source": "Experimental"
            })
experimental_df = pd.DataFrame(experimental_df)

# Print experimental_df to verify values
print("Experimental DataFrame (mean values):")
print(experimental_df.groupby(["Tissue", "Species", "Source"])["Concentration (mg/L)"].mean())

# Combine model and experimental data for steady-state comparison
all_steady_data = pd.concat([chicken_steady, duck_steady, experimental_df], ignore_index=True)
all_steady_data["Source_Species"] = all_steady_data["Source"] + " (" + all_steady_data["Species"] + ")"

# Print all_steady_data to verify values
print("All Steady-State DataFrame (mean values):")
print(all_steady_data.groupby(["Tissue", "Source_Species"])["Concentration (mg/L)"].mean())

# Calculate standard deviations for better insight
print("All Steady-State DataFrame (standard deviations):")
print(all_steady_data.groupby(["Tissue", "Source_Species"])["Concentration (mg/L)"].std())

# Set Seaborn style for a formal look
sns.set_style("whitegrid")
sns.set_palette("muted")

# Create the subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Subplot 1: Predicted PFOA concentration profiles over 700 hours with multi-dose
sns.lineplot(data=line_df, x="Time (h)", y="Concentration (mg/L)", hue="Tissue", style="Tissue", 
             ci="sd", ax=axes[0])
for dose_time in dose_times[:-1]:
    axes[0].axvline(x=dose_time, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title("Predicted PFOA Concentration Profiles with Multi-Dose (0.0005 mg every 24 h for 7 Days)", fontsize=16)
axes[0].set_xlabel("Time (hours)", fontsize=14)
axes[0].set_ylabel("Concentration (mg/L)", fontsize=14)
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].legend(title="Tissue", fontsize=10)

# Subplot 2: Steady-state comparison with error bars for all bars
# Define hue order to match legend
hue_order = ["Model (Chicken)", "Model (Duck)", "Experimental (Chicken)", "Experimental (Duck)"]

# Plot all bars without automatic error bars, adjust bar width
bar_plot = sns.barplot(data=all_steady_data, x="Tissue", y="Concentration (mg/L)", hue="Source_Species", 
                       hue_order=hue_order, ci=None, ax=axes[1], width=0.8)

# Calculate means and standard deviations for all data
stats = all_steady_data.groupby(["Tissue", "Source_Species"])["Concentration (mg/L)"].agg(['mean', 'std']).reset_index()

# Add custom error bars for all bars
tissues = ["Liver", "Meat", "Eggs"]
bar_width = bar_plot.patches[0].get_width()
group_width = bar_width * len(hue_order)

for i, tissue in enumerate(tissues):
    for j, hue in enumerate(hue_order):
        # Calculate the x-position of the bar
        x_pos = i + (j - (len(hue_order) - 1) / 2) * bar_width
        stats_row = stats[(stats["Tissue"] == tissue) & (stats["Source_Species"] == hue)]
        if not stats_row.empty:
            mean = stats_row["mean"].values[0]
            std = stats_row["std"].values[0]
            axes[1].errorbar(x_pos, mean, yerr=std, fmt='none', color='black', capsize=3)

# Add hatch patterns to distinguish Chicken vs. Duck
hatch_dict = {
    "Model (Chicken)": '//',
    "Model (Duck)": '',
    "Experimental (Chicken)": '//',
    "Experimental (Duck)": ''
}

# Debug the number of bars and their properties
print(f"Number of bars in bar_plot.patches: {len(bar_plot.patches)}")
print(f"Expected number of bars: {len(tissues) * len(hue_order)}")

# Create a mapping of x-positions to hue categories
bar_info = []
for i, bar in enumerate(bar_plot.patches):
    x_pos = bar.get_x() + bar.get_width() / 2
    tissue_idx = int(x_pos // 1)  # Integer part of x_pos determines the tissue group
    bar_info.append({
        "index": i,
        "x_pos": x_pos,
        "tissue_idx": tissue_idx,
        "color": bar.get_facecolor()
    })

# Map colors to hue categories (based on the order in the legend)
legend = axes[1].get_legend()
try:
    # For Matplotlib >= 3.7
    legend_colors = [t.get_facecolor() for t in legend.legend_handles]
except AttributeError:
    # For Matplotlib < 3.7
    legend_colors = [t.get_facecolor() for t in legend.get_legend_handles()]
color_to_hue = {tuple(legend_colors[i]): hue for i, hue in enumerate(hue_order)}

# Apply hatch patterns based on the hue determined by the bar's color
for info in bar_info:
    i = info["index"]
    tissue_idx = info["tissue_idx"]
    color = info["color"]
    hue = color_to_hue[tuple(color)]
    bar = bar_plot.patches[i]
    bar.set_hatch(hatch_dict[hue])
    tissue_name = tissues[tissue_idx] if tissue_idx < len(tissues) else "Unknown"
    print(f"Bar {i} (Tissue: {tissue_name}, Hue: {hue}, Color: {color}) - Hatch: {hatch_dict[hue]}")

# Add MRL lines for each tissue
axes[1].axhline(y=0.001, color='red', linestyle='--', label='MRL Liver (0.001 mg/L)', alpha=0.5)
axes[1].axhline(y=0.0003, color='blue', linestyle='--', label='MRL Meat (0.0003 mg/L)', alpha=0.5)
axes[1].axhline(y=0.0001, color='green', linestyle='--', label='MRL Eggs (0.0001 mg/L)', alpha=0.5)

# Customize the bar plot
axes[1].set_title("PBPK Model Predictions vs Experimental Data at Steady State (700 hours)", fontsize=16)
axes[1].set_xlabel("Tissue", fontsize=14)
axes[1].set_ylabel("Concentration (mg/L)", fontsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# Set y-axis limit to zoom in on the data
axes[1].set_ylim(0, 0.0004)  # Adjusted to focus on the data range (max concentration ~0.00025 mg/L)

# Move the legend outside the plot to avoid clutter
axes[1].legend(title="Source (Species) / MRL", fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("Figure1_corrected.png", dpi=300, bbox_inches='tight')
print("Saved Figure1_corrected.png")