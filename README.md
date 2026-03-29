# JT Gravity from Counting Causally Convex Subsets

**Paper IV** of the Causal-Algebraic Geometry series.

## Summary

We define a partition function over causally convex subsets of a discrete spacetime, weighted by the Benincasa-Dowker action: Z(β) = Σ exp(−β·S_BD[S]). On variable-width 2D grids, the effective action extracted from Z(β) correlates with the structure of Jackiw-Teitelboim gravity: a topological term (discrete curvature R) and a conformal-mode kinetic term (the log-Schwarzian).

At n = 15 (42 profiles): R² = 0.85 for the two-parameter fit (R + LogSch), with both coefficients scaling linearly with β (R² > 0.99).

At n = 30 (50 profiles): the topological R term becomes negligible (R² < 0.01) while the log-Schwarzian retains R² = 0.65 with β-stable coefficients (R² > 0.99). This is consistent with the expected 2D behavior: the Einstein-Hilbert action is topological (Gauss-Bonnet) and its finite-size contribution washes out at larger n, leaving the JT/Schwarzian as the sole dynamical term.

The action and entropy contributions are nearly orthogonal: R controls the saddle-point action (R² = 0.19), LogSch controls the fluctuation entropy (R² = 0.69), and the cross-term is small (R² = 0.10 for entropy vs R).

## Key Results

| Result | Value | Source |
|--------|-------|--------|
| JT fit R² (n=15) | 0.846 (β=1), 0.902 (β=3) | `04_jt_model_comparison.py` |
| JT fit R² (n=30) | 0.653 (β=1), LogSch dominates | `11_n30_10row_validation.py` |
| β-linearity of coefficients | R² > 0.99 at both n=15 and n=30 | `05_beta_scaling.py` |
| Action/entropy orthogonality | Entropy vs R: R² = 0.10 | `06_action_entropy_decomposition.py` |
| JT entropy S(β) = a/√β + S₀ | R² = 0.987 | `07_thermodynamics_and_dos.py` |
| Density of states | Gaussian envelope R² = 0.996 | `07_thermodynamics_and_dos.py` |
| d-dependence: fixed-N R² | d=2: 0.35 → d=3: 0.51 (preliminary, small sample) | Not scripted |
| UV cutoff | S_BD,max = min(d, m) = Dilworth width | Universal |
| S₄ check (d=4) | 2 dominant eigenvalues = S₄ irrep, not gravitons | `02_bd_4d_graviton_test.py` |

## Scripts

All scripts are self-contained Python files. Run with `python3 scripts/XX_name.py`.

### Core JT gravity analysis (Paper IV)

| Script | Description |
|--------|-------------|
| `01_bd_partition_function.py` | BD action and partition function Z(β) on [m]² for m=2,3,4 |
| `02_bd_4d_graviton_test.py` | Fluctuation spectrum on [2]⁴, S₄ representation analysis |
| `03_controlled_eh_test.py` | 42 variable-width grids at n=15, R-only fit (R²=0.27) |
| `04_jt_model_comparison.py` | 10 models compared: R+LogSch gives R²=0.846 |
| `05_beta_scaling.py` | Coefficients a(β), b(β) at 8 values of β |
| `06_action_entropy_decomposition.py` | Action→R, entropy→LogSch, orthogonality check |
| `07_thermodynamics_and_dos.py` | Density of states, entropy S(β), specific heat C(β) |
| `08_curved_vs_flat.py` | Initial curved-vs-flat comparison (6 grids, 4 parameters — underpowered) |
| `09_hessian_analysis.py` | Hessian of S_BD at the flat saddle point |

### n=30 validation

| Script | Description |
|--------|-------------|
| `10_n30_5row_validation.py` | n=30, 5-row grids: R²=0.69 for R+LogSch |
| `11_n30_10row_validation.py` | n=30, 10-row grids: R²=0.65, R washes out to <0.01 |

### Exploratory (not in Paper IV)

| Script | Description |
|--------|-------------|
| `12_rna_partition_function.py` | Convex subset partition function on RNA base-pair posets; boundary dominance confirmed on synthetic structures |
| `13_substance_audit.lean` | Lean theorem type verification |
| `14_rfam_gamma.py` | Noetherian ratio γ on 14 Rfam families (exact, base-pair containment poset) |
| `15_shape_validation_literature.py` | ⟨φ⟩ vs literature SHAPE values (r=−0.89 per-structure mean; representative values, not raw data) |
| `16_shape_validation_vienna.py` | ⟨φ⟩ vs ViennaRNA bp probabilities (r=0.08 pooled — negative result, non-circular test fails) |
| `17_reaction_network_gamma.py` | γ on metabolic reaction networks (E. coli central carbon γ=1.59) |
| `18_action_entropy_decomposition.py` | Duplicate of 06 (action–entropy separation) |
| `19_thermodynamics_dos.py` | Duplicate of 07 (thermodynamics) |

## Dependencies

- Python 3.8+
- NumPy
- ViennaRNA Python module (optional, for script 16 only)

## Companion Repositories

- [causal-algebraic-geometry-lean](https://github.com/tomdif/causal-algebraic-geometry-lean) — Lean 4 formalization (dimension law, tiling inequality, ρ₂ = 16 fully proved, Wilson loop)
- Papers I–III: Grid Convex Subsets, Causal-Algebraic Geometry, Black Hole Thermodynamics

## Citation

```
@article{DiFiore2026JT,
  title={Jackiw-Teitelboim Gravity and Hagedorn Density of States from Counting Causally Convex Subsets},
  author={DiFiore, Thomas},
  year={2026}
}
```
