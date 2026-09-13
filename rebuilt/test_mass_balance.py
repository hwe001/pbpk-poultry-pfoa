#!/usr/bin/env python3
"""Mass-balance verification for the rebuilt poultry PFOA PBPK model.

The defining failure of the original scripts was structural non-conservation.
This test asserts the rebuilt model conserves mass to solver tolerance across
regimens that exercise every pathway: single bolus, the 7-day legacy course,
a long horizon (post-dose washout), and a heavy repeated-dose case, for both
species. Run:  python test_mass_balance.py
"""

from __future__ import annotations

import numpy as np

from pbpk_poultry_rebuilt import chicken_params, duck_params, simulate

TOL = 1e-6  # fractional conservation error allowed (solver tolerance ~1e-8/step)


def check(name: str, sol) -> None:
    status = "PASS" if sol.mass_err < TOL else "FAIL"
    print(f"[{status}] {name:38s} mass_err = {sol.mass_err:.2e}  "
          f"(dose={sol.dose_total:.4g} ug/kg)")
    assert sol.mass_err < TOL, f"mass balance violated: {sol.mass_err:.3e}"


def main() -> None:
    for params in (chicken_params(), duck_params()):
        p = params.name
        # 1. Single oral bolus, short horizon
        check(f"{p}: single bolus, 168 h",
              simulate(params, [0.0], [1.0], t_end=168.0))
        # 2. Legacy 7-day course, 700 h horizon (dosing ends at 144 h -> washout)
        check(f"{p}: 7 daily boluses, 700 h",
              simulate(params, np.arange(0.0, 145.0, 24.0), np.full(7, 0.5), t_end=700.0))
        # 3. Long repeated-dose case, 30 days
        check(f"{p}: 30 daily boluses, 720 h",
              simulate(params, np.arange(0.0, 697.0, 24.0), np.full(30, 2.0), t_end=720.0))
        # 4. Zero-dose run (must stay at exactly zero, no mass invented)
        sol = simulate(params, [], [], t_end=100.0)
        assert np.abs(sol.amounts).max() < 1e-15, "zero-dose run generated mass"
        print(f"[PASS] {p}: zero-dose run stays at zero")

    print("\nAll mass-balance checks passed.")


if __name__ == "__main__":
    main()
