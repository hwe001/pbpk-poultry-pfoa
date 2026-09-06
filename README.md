# pbpk-poultry-pfoa

Supplementary code for:

> Ho, Tang, Bai, Zhang. "Physiologically Based Pharmacokinetic Modeling of PFOA
> in Poultry: Insights into Tissue Distribution."

An 8-compartment physiologically based pharmacokinetic (PBPK) model
simulating PFOA (perfluorooctanoic acid) kinetics and tissue distribution in
chickens and ducks, calibrated against physiological parameters from Wang et
al. (2021) and validated against environmental exposure data from Chongqing,
China (Tang et al., 2025).

## Contents

- **`pbpk_poultry_model.py`** — the core model: 8 compartments (Gut, Plasma,
  Liver, Kidney, Muscle, Fat, Eggs, Rest of Body), a 7-day multi-dose
  regimen (0.5 µg/24h), simulated to 700h. Runs a 500-iteration Monte Carlo
  uncertainty analysis (±30% on urinary clearance and liver partition
  coefficient) and plots predicted concentration profiles alongside the
  Chongqing experimental data and regulatory Maximum Residue Limits.
- **`pbpk_poultry_local_sensitivity.py`** — local sensitivity analysis:
  perturbs each of 6 parameters (Ka, K_urine, K_met, P_liver, P_muscle,
  P_egg) by +10% one at a time and reports the resulting sensitivity index
  (%change in tissue concentration / %change in parameter) for liver,
  muscle, and egg.
- **`pbpk_poultry_global_sensitivity_sobol.py`** — global (Sobol) sensitivity
  analysis over 4 parameters (K_urine, P_liver, P_muscle, P_egg) at ±30%
  ranges, 1024 Saltelli base samples (6144 model evaluations per species),
  with first-order (S1) and total-order (ST) indices plus real bootstrap
  95% confidence intervals (not the naive point-estimate bars an earlier
  version of this analysis used).

All three scripts are self-contained (no external data files) and write
their output plots/CSVs to the working directory.

## Running

```bash
pip install -r requirements.txt
python pbpk_poultry_model.py                       # ~1 min
python pbpk_poultry_local_sensitivity.py           # ~1 min
python pbpk_poultry_global_sensitivity_sobol.py    # ~30-40 min (12,288 stiff ODE solves)
```

## A note on reproducibility

An earlier version of the global sensitivity script (used to produce this
paper's original figures) had two bugs, caught and fixed during manuscript
revision:

1. It used a stale, differently-calibrated parameter set (absolute L/h flow
   values and an outdated liver partition coefficient) rather than the
   values that actually produced the paper's validated Figure 1 results, and
   a dose 1000x too large.
2. It ran far fewer samples than the manuscript's stated methodology (1000
   or 10,000 total simulations depending on version, rather than the stated
   6144 per species), using ±15-20% ranges instead of the stated ±20-40%.

The version in this repository is the corrected one, verified to use
exactly the parameters in `pbpk_poultry_model.py` and the sample size/ranges
described in the manuscript's Methods.

The local sensitivity script was similarly rewritten: the version used to
produce the original figures only plotted a hardcoded numeric table with no
underlying simulation code to verify or reproduce it. The version here
recomputes those values directly from the model equations via parameter
perturbation, and reproduces the manuscript's reported sensitivity ranges
closely.

## Requirements

See `requirements.txt`. Tested with the versions listed there; SALib's
`saltelli` sampler is deprecated in favor of `salib.sample.sobol` as of
SALib 1.5 but both scripts still work with the deprecated call.

## Contact

Dr Harvey Ho, Auckland Bioengineering Institute, University of Auckland —
harvey.ho@auckland.ac.nz
