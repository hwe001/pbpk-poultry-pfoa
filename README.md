# pbpk-poultry-pfoa

Supplementary code for:

> Ho, Tang, Bai, Zhang. "Physiologically based pharmacokinetic modelling of
> PFOA in laying poultry: reverse-dosimetry estimation of feed exposure from
> survey tissue concentrations."

An 8-compartment PBPK model of PFOA (perfluorooctanoic acid) in chickens and
ducks, calibrated against survey tissue concentrations from Chongqing, China
(Tang et al., 2024).

## Current state of this repository (2026-09)

The original model implementation contained structural defects that made its
quantitative results unreliable. During manuscript revision the model was
rebuilt from first principles; the rebuilt code is now the authoritative
implementation, and the original scripts are retained under `legacy/` for
provenance only.

### Why the legacy scripts were retired (`legacy/`)

- Renal clearance was applied twice (a `-K_urine` sink in **both** the plasma
  and the kidney equations), so every fitted clearance value absorbed the
  duplication.
- "Flows" were dimensionless fractions of cardiac output used as flow *rates*;
  there were no tissue volumes and no cardiac output, so the equations could
  not conserve mass.
- The three scripts implemented **different** gut-absorption pathways
  (gut→plasma in the main model, gut→liver in both sensitivity scripts), so
  no published result could be reproduced from the repository.
- The dosing code added an amount (mg) directly into a concentration state.
- The "experimental" error bars in the figure code were synthetic (±5%
  normal noise generated around published means), not measured variability.

### The rebuilt model (`rebuilt/`)

Standard flow-limited, well-stirred PBPK formulation: state variables are
amounts (µg/kg body weight), tissues have volumes and blood flows in L/h,
absorbed dose enters the liver as portal input, renal clearance is applied
exactly once, no metabolism term (PFOA is not metabolised), and the egg
compartment loses mass at the oviposition rate (~1 egg/25 h) with laid-egg
mass tracked as an excretion route. Body-weight-normalised units throughout;
every non-literature parameter is flagged `# ASSUMPTION` in the source.

- **`rebuilt/pbpk_poultry_rebuilt.py`** — model core + analytic and numerical
  machinery.
- **`rebuilt/test_mass_balance.py`** — asserts mass conservation (dose in =
  stored + renally excreted + oviposited) to ~1e-15 relative error across
  regimens and species. The legacy code could not pass such a test.
- **`rebuilt/run_rebuilt_demo.py`** — legacy 7-day scenario demonstration and
  preliminary reduced refit. Its output is *diagnostic*: matching the survey
  means at a 700-h post-dosing snapshot forces an elimination half-life of
  4–7 h, ~2 orders of magnitude faster than PFOA's known persistence —
  evidence that the post-dosing scenario itself was incompatible with the
  data, not merely the parameters.
- **`rebuilt/reverse_dosimetry.py`** — the analysis the revised manuscript
  uses: analytic steady state under continuous exposure, back-calculation of
  the exposure required to explain the survey means (with Monte Carlo
  uncertainty and variance-based sensitivity indices implemented and
  self-tested in-file; SALib not required), and translation to feed/water
  concentrations.

## Status caveats

- The calibration-target unit basis (µg/L vs µg/kg wet weight) is pending
  confirmation against the source survey; the pattern-based conclusions are
  invariant to it, the absolute back-calculated exposures are not.
- Physiological volumes/flows carry `# ASSUMPTION` flags pending verification
  against Wang et al. (2021); no reliable poultry-specific PFOA half-life was
  identified, so it is swept (0.5–14 d) rather than assumed silently.
- Figures in `results/` are preliminary working outputs, not publication
  figures.

## Running

```bash
pip install numpy scipy matplotlib
python rebuilt/test_mass_balance.py      # mass-balance verification (~s)
python rebuilt/run_rebuilt_demo.py       # legacy-scenario diagnostic (~1 min)
python rebuilt/reverse_dosimetry.py      # steady-state reverse dosimetry (~10 s)
```

## Contact

Dr Harvey Ho — harvey@ratalab.nz
