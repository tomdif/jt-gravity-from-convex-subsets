# JT Gravity from Counting Causally Convex Subsets

**Paper IV** of the Causal-Algebraic Geometry series.

## Summary

We show that the Benincasa-Dowker (BD) weighted partition function over causally convex subsets of discrete spacetimes reproduces the Jackiw-Teitelboim (JT) gravity effective action in 2D. The decomposition

$$\log Z(\beta) = -\beta \cdot (a_0 \cdot R + b_0 \cdot \mathrm{LogSch}) + \text{entropy}$$

emerges from 42 variable-width 2D grids at fixed element count $n=15$, with:
- $R^2 = 0.846$ for the two-parameter fit (R + LogSchwarzian)
- Both coefficients scaling linearly with $\beta$ ($R^2 > 0.99$)
- A clean **action–entropy decomposition**: R controls the saddle-point action ($R^2=0.19$), LogSch controls the fluctuation entropy ($R^2=0.69$), and the two are orthogonal ($R^2=0.10$ for entropy vs R)

## Key Results

| Result | Value | Source |
|--------|-------|--------|
| JT fit $R^2$ | 0.846 (β=1), 0.902 (β=3) | `04_jt_model_comparison.py` |
| β-linearity of coefficients | $R^2 > 0.99$ | `05_beta_scaling.py` |
| Action/entropy orthogonality | Entropy vs R: $R^2 = 0.10$ | `06_action_entropy_decomposition.py` |
| JT entropy $S(\beta) = a/\sqrt{\beta} + S_0$ | $R^2 = 0.987$ | `07_thermodynamics_and_dos.py` |
| Density of states | Gaussian envelope $R^2 = 0.996$ | `07_thermodynamics_and_dos.py` |
| d-dependence: fixed-N $R^2$ | d=2: 0.35 → d=3: 0.51 | Agent computation |
| UV cutoff | $S_{\mathrm{BD}}^{\max} = \min(d, m)$ = Dilworth width | Universal |
| S₄ check (d=4) | 2 dominant eigenvalues = S₄ irrep, not gravitons | `02_bd_4d_graviton_test.py` |

## Scripts

All scripts are self-contained Python files. Run with `python3 scripts/XX_name.py`.

| Script | Description |
|--------|-------------|
| `01_bd_partition_function.py` | BD action and partition function Z(β) on [m]² for m=2,3,4 |
| `02_bd_4d_graviton_test.py` | Fluctuation spectrum on [2]⁴, S₄ representation analysis |
| `03_controlled_eh_test.py` | 42 variable-width grids, R-only fit (R²=0.27) |
| `04_jt_model_comparison.py` | 10 models compared: R+LogSch wins at R²=0.846 |
| `05_beta_scaling.py` | Coefficients a(β), b(β) at 8 values of β |
| `06_action_entropy_decomposition.py` | Analytical derivation: action→R, entropy→LogSch |
| `07_thermodynamics_and_dos.py` | Density of states, entropy S(β), specific heat C(β) |
| `08_curved_vs_flat.py` | Initial curved-vs-flat comparison (6 grids) |
| `09_hessian_analysis.py` | Hessian of S_BD at the flat saddle point |

## Dependencies

- Python 3.8+
- NumPy
- No other dependencies (all computations are self-contained brute-force enumerations)

## Companion Repositories

- [causal-algebraic-geometry-lean](https://github.com/tomdif/causal-algebraic-geometry-lean) — Lean 4 formalization (dimension law, tiling inequality, Wilson loop)
- Papers I–III: Grid Convex Subsets, Causal-Algebraic Geometry, Black Hole Thermodynamics

## Citation

```
@article{DiFiore2026JT,
  title={Jackiw-Teitelboim Gravity from Counting Causally Convex Subsets},
  author={DiFiore, Thomas},
  year={2026}
}
```
