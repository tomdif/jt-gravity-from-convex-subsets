"""
Controlled test: fix n, vary R, check if log Z is linear in R.

Strategy: generate grids with EXACTLY n=15 elements and 5 rows,
but different width profiles giving different curvatures.
Width profiles summing to 15 with 5 rows: w1+w2+w3+w4+w5 = 15, each wi >= 1.
"""
import numpy as np
from collections import defaultdict
import math
from itertools import combinations_with_replacement

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
            results.append((len(S) - links, len(S), S))
    return results

def discrete_curvature(widths):
    R = 0
    for i in range(1, len(widths)-1):
        R += widths[i-1] + widths[i+1] - 2*widths[i]
    return R

# Generate ALL width profiles with 5 rows summing to 15, each row >= 1, max width <= 7
# (max 7 to keep 2^15 = 32768 brute force feasible)
profiles = set()
for w1 in range(1, 8):
    for w2 in range(1, 8):
        for w3 in range(1, 8):
            for w4 in range(1, 8):
                w5 = 15 - w1 - w2 - w3 - w4
                if 1 <= w5 <= 7:
                    profiles.add((w1, w2, w3, w4, w5))

print(f"Total profiles with n=15, 5 rows: {len(profiles)}")

# Group by curvature
by_R = defaultdict(list)
for p in profiles:
    R = discrete_curvature(p)
    by_R[R].append(p)

print(f"Curvature values: {sorted(by_R.keys())}")
for R in sorted(by_R.keys()):
    print(f"  R={R:>3}: {len(by_R[R])} profiles")

# Select ~20 diverse profiles spanning the curvature range
# Pick 1-2 from each curvature value
selected = []
for R in sorted(by_R.keys()):
    profs = sorted(by_R[R])
    selected.append(profs[0])  # first
    if len(profs) > 1:
        selected.append(profs[len(profs)//2])  # middle

print(f"\nSelected {len(selected)} profiles for computation")

# Compute log Z for each at beta = 1.0
beta = 1.0
results = []
for i, widths in enumerate(selected):
    R = discrete_curvature(widths)
    cc = enumerate_cc_varwidth(widths)
    sbd_arr = np.array([c[0] for c in cc], dtype=float)
    w = np.exp(-beta * sbd_arr)
    Z = w.sum()
    logZ = math.log(Z)
    results.append((widths, R, len(cc), logZ))
    if (i+1) % 5 == 0:
        print(f"  Computed {i+1}/{len(selected)}...")

print(f"\nAll {len(selected)} computed.")

# Sort by R
results.sort(key=lambda x: x[1])

print(f"\n{'Profile':>20} {'R':>4} {'|CC|':>8} {'log Z':>10}")
print("-" * 50)
for widths, R, ncc, logZ in results:
    print(f"{str(widths):>20} {R:>4} {ncc:>8} {logZ:>10.4f}")

# Fit: log Z = a + b*R (linear only, since n is fixed)
R_arr = np.array([r[1] for r in results], dtype=float)
logZ_arr = np.array([r[3] for r in results], dtype=float)

# Linear fit
A_lin = np.column_stack([R_arr, np.ones(len(results))])
coeffs_lin = np.linalg.lstsq(A_lin, logZ_arr, rcond=None)[0]
res_lin = logZ_arr - A_lin @ coeffs_lin
rms_lin = np.sqrt(np.mean(res_lin**2))

# Quadratic fit
A_quad = np.column_stack([R_arr, R_arr**2, np.ones(len(results))])
coeffs_quad = np.linalg.lstsq(A_quad, logZ_arr, rcond=None)[0]
res_quad = logZ_arr - A_quad @ coeffs_quad
rms_quad = np.sqrt(np.mean(res_quad**2))

print(f"\n{'='*50}")
print(f"CONTROLLED FIT: n=15 fixed, {len(results)} data points")
print(f"{'='*50}")
print(f"\nLinear fit: log Z = {coeffs_lin[0]:.4f}·R + {coeffs_lin[1]:.4f}")
print(f"  RMS residual = {rms_lin:.4f}")
print(f"  R² = {1 - np.var(res_lin)/np.var(logZ_arr):.4f}")

print(f"\nQuadratic fit: log Z = {coeffs_quad[0]:.4f}·R + {coeffs_quad[1]:.5f}·R² + {coeffs_quad[2]:.4f}")
print(f"  RMS residual = {rms_quad:.4f}")
print(f"  R² = {1 - np.var(res_quad)/np.var(logZ_arr):.4f}")

print(f"\nImprovement from R²: RMS {rms_lin:.4f} → {rms_quad:.4f} ({(1-rms_quad/rms_lin)*100:.1f}% reduction)")

# Is log Z actually linear in R?
# Check: plot-like output
print(f"\nResiduals from linear fit:")
for widths, R, ncc, logZ in results:
    fitted = coeffs_lin[0] * R + coeffs_lin[1]
    print(f"  R={R:>3}: logZ={logZ:.3f}, fit={fitted:.3f}, res={logZ-fitted:>+.3f}")
