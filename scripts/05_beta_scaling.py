"""
β-scaling of the JT gravity coefficients.

If the decomposition log Z(β) = a(β)·R + b(β)·LogSch + c(β) is physical,
then a(β) and b(β) should scale LINEARLY with β:
  a(β) = β·a₀,  b(β) = β·b₀
because log Z = -β·<S_BD> + entropy, and at large β the action dominates.

Test: fit the R + LogSch model at β = 0.5, 1.0, 1.5, 2.0, 3.0
and check if coefficients scale linearly.
"""
import numpy as np
from collections import defaultdict
import math

def enumerate_cc_varwidth(row_widths):
    n_rows = len(row_widths)
    cells = [(r,c) for r in range(n_rows) for c in range(row_widths[r])]
    n = len(cells)
    cell_set = set(cells)
    results = []
    for bits in range(1 << n):
        S = frozenset(cells[k] for k in range(n) if bits & (1 << k))
        convex = True
        for a in S:
            if not convex: break
            for b in S:
                if not convex: break
                if a[0] <= b[0] and a[1] <= b[1]:
                    for r in range(a[0], b[0]+1):
                        for c in range(a[1], b[1]+1):
                            if (r,c) in cell_set and (r,c) not in S:
                                convex = False; break
                        if not convex: break
        if convex:
            links = sum(1 for (r,c) in S if (r+1,c) in S) + \
                    sum(1 for (r,c) in S if (r,c+1) in S)
            results.append(len(S) - links)
    return results

def compute_log_schwarzian(widths):
    w = np.array(widths, dtype=float)
    return sum((math.log(w[i+1]) - math.log(w[i]))**2 for i in range(len(w)-1))

def compute_R(widths):
    return sum(widths[i-1] + widths[i+1] - 2*widths[i] for i in range(1, len(widths)-1))

# Generate profiles (same as before)
profiles = set()
for w1 in range(1, 8):
    for w2 in range(1, 8):
        for w3 in range(1, 8):
            for w4 in range(1, 8):
                w5 = 15 - w1 - w2 - w3 - w4
                if 1 <= w5 <= 7:
                    profiles.add((w1, w2, w3, w4, w5))

by_R = defaultdict(list)
for p in profiles:
    by_R[compute_R(p)].append(p)

selected = []
for R in sorted(by_R.keys()):
    profs = sorted(by_R[R])
    selected.append(profs[0])
    if len(profs) > 1:
        selected.append(profs[len(profs)//2])

print(f"Computing {len(selected)} profiles at multiple β values...")

# Precompute SBD distributions for each profile
profile_data = []
for i, widths in enumerate(selected):
    R = compute_R(widths)
    LS = compute_log_schwarzian(widths)
    sbd_list = enumerate_cc_varwidth(widths)
    sbd_arr = np.array(sbd_list, dtype=float)
    profile_data.append((widths, R, LS, sbd_arr))
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(selected)}")

print("Done.\n")

R_arr = np.array([d[1] for d in profile_data])
LS_arr = np.array([d[2] for d in profile_data])
n_pts = len(profile_data)

# Fit at each β
betas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
print(f"{'β':>6} {'a(R)':>10} {'b(LogSch)':>10} {'c(const)':>10} {'R²':>8} {'a/β':>8} {'b/β':>8}")
print("-" * 70)

a_vals = []
b_vals = []
r2_vals = []

for beta in betas:
    logZ = np.array([math.log(np.exp(-beta * d[3]).sum()) for d in profile_data])
    
    A = np.column_stack([R_arr, LS_arr, np.ones(n_pts)])
    coeffs = np.linalg.lstsq(A, logZ, rcond=None)[0]
    fitted = A @ coeffs
    res = logZ - fitted
    r2 = 1 - np.var(res) / np.var(logZ)
    
    a_vals.append(coeffs[0])
    b_vals.append(coeffs[1])
    r2_vals.append(r2)
    
    a_over_beta = coeffs[0] / beta if beta > 0 else 0
    b_over_beta = coeffs[1] / beta if beta > 0 else 0
    
    print(f"{beta:>6.2f} {coeffs[0]:>10.4f} {coeffs[1]:>10.4f} {coeffs[2]:>10.4f} {r2:>8.4f} {a_over_beta:>8.4f} {b_over_beta:>8.4f}")

# Linear fit of a(β) and b(β) vs β
beta_arr = np.array(betas)
a_arr = np.array(a_vals)
b_arr = np.array(b_vals)

# a(β) = a₀·β + a₁
a_fit = np.polyfit(beta_arr, a_arr, 1)
b_fit = np.polyfit(beta_arr, b_arr, 1)

# R² of the linear fit
a_pred = np.polyval(a_fit, beta_arr)
b_pred = np.polyval(b_fit, beta_arr)
a_r2 = 1 - np.var(a_arr - a_pred) / np.var(a_arr)
b_r2 = 1 - np.var(b_arr - b_pred) / np.var(b_arr)

print(f"\n{'='*60}")
print(f"β-SCALING ANALYSIS")
print(f"{'='*60}")
print(f"\na(β) = {a_fit[0]:.4f}·β + {a_fit[1]:.4f}  (R² = {a_r2:.4f})")
print(f"b(β) = {b_fit[0]:.4f}·β + {b_fit[1]:.4f}  (R² = {b_r2:.4f})")
print(f"\nRatio a(β)/b(β) at each β:")
for i, beta in enumerate(betas):
    if b_vals[i] != 0:
        print(f"  β={beta:.2f}: a/b = {a_vals[i]/b_vals[i]:.4f}")

ratio_std = np.std([a_vals[i]/b_vals[i] for i in range(len(betas)) if b_vals[i] != 0])
ratio_mean = np.mean([a_vals[i]/b_vals[i] for i in range(len(betas)) if b_vals[i] != 0])
print(f"\n  Mean a/b = {ratio_mean:.4f}")
print(f"  Std  a/b = {ratio_std:.4f}")
print(f"  Stability: {'STABLE' if ratio_std / abs(ratio_mean) < 0.1 else 'UNSTABLE'} ({ratio_std/abs(ratio_mean)*100:.1f}% variation)")

print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
if a_r2 > 0.95 and b_r2 > 0.95:
    print(f"Both coefficients scale linearly with β (R² > 0.95).")
    print(f"The decomposition log Z = β·(a₀·R + b₀·LogSch) + entropy")
    print(f"is CONFIRMED as physical, with:")
    print(f"  a₀ = {a_fit[0]:.4f} (EH/topological coefficient)")
    print(f"  b₀ = {b_fit[0]:.4f} (JT/Schwarzian coefficient)")
    print(f"  Ratio a₀/b₀ = {a_fit[0]/b_fit[0]:.4f}")
else:
    print(f"Coefficients do NOT scale linearly with β.")
    print(f"  a(β) linearity: R² = {a_r2:.4f}")
    print(f"  b(β) linearity: R² = {b_r2:.4f}")
