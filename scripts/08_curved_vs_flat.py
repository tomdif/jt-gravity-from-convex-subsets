"""
THE SHOWSTOPPER: BD partition function on CURVED vs FLAT grids.

If log Z_BD(curved) - log Z_BD(flat) = α · R · Area + O(1),
where R is the discrete Ricci scalar, then the BD partition function
IS the discrete Einstein-Hilbert action on convex subsets.

Test: compare grids with same total elements but different curvature profiles.
"""
import numpy as np
from collections import defaultdict
import math

def enumerate_cc_variable_width(row_widths):
    """Enumerate all convex subsets of a variable-width 2D grid.
    Row i has width row_widths[i]. Cells: (row, col) with 0 ≤ col < row_widths[row].
    Product order: (r1,c1) ≤ (r2,c2) iff r1≤r2 and c1≤c2."""
    n_rows = len(row_widths)
    cells = [(r,c) for r in range(n_rows) for c in range(row_widths[r])]
    n = len(cells)
    cell_set = set(cells)
    
    results = []
    for bits in range(1 << n):
        S = frozenset(cells[k] for k in range(n) if bits & (1 << k))
        
        # Check convexity
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
                            # If (r,c) not in grid at all: need to check
                            # Actually for variable-width grids, (r,c) might not exist
                            # Convexity only requires cells that EXIST in the grid
                            if (r,c) not in cell_set:
                                # Cell doesn't exist — is this a problem?
                                # For product-order convexity on the SUBPOSET:
                                # we only check cells that are in the poset
                                pass
                        if not convex: break
        
        if convex:
            # Compute S_BD: |S| - links
            links = 0
            for (r,c) in S:
                if (r+1,c) in S: links += 1
                if (r,c+1) in S: links += 1
            sbd = len(S) - links
            results.append((sbd, len(S), S))
    
    return results

def grid_stats(row_widths, name):
    """Compute BD partition function statistics for a variable-width grid."""
    n_total = sum(row_widths)
    results = enumerate_cc_variable_width(row_widths)
    n_cc = len(results)
    
    sbd_arr = np.array([r[0] for r in results], dtype=float)
    size_arr = np.array([r[1] for r in results], dtype=float)
    
    # Discrete curvature: second derivative of row widths
    n_rows = len(row_widths)
    curvature = 0
    for i in range(1, n_rows-1):
        R_i = row_widths[i-1] + row_widths[i+1] - 2*row_widths[i]
        curvature += R_i
    
    print(f"\n{'='*50}")
    print(f"{name}: widths={row_widths}, n={n_total}, |CC|={n_cc}")
    print(f"Discrete curvature Σ R_i = {curvature}")
    print(f"  (R > 0 = sphere/positive, R < 0 = saddle/negative, R = 0 = flat)")
    
    # Partition function at various β
    print(f"\n{'β':>6} {'log Z':>10} {'<S_BD>':>8} {'<|S|>':>8}")
    log_Z = {}
    for beta in [0.5, 1.0, 2.0, 3.0]:
        w = np.exp(-beta * sbd_arr)
        Z = w.sum()
        avg_s = (sbd_arr * w).sum() / Z
        avg_n = (size_arr * w).sum() / Z
        log_Z[beta] = math.log(Z)
        print(f"{beta:>6.1f} {math.log(Z):>10.4f} {avg_s:>8.3f} {avg_n:>8.3f}")
    
    return n_cc, curvature, log_Z

# Test grids with same total elements but different curvature
# All have 5 rows, total ~15 elements
print("CURVED vs FLAT BD PARTITION FUNCTION")
print("="*50)
print("Grids with ~same total elements, different curvature")

configs = [
    ("Flat",    [3, 3, 3, 3, 3]),       # R = 0
    ("Sphere",  [2, 3, 4, 3, 2]),       # R > 0 (wide middle)
    ("Saddle",  [4, 3, 2, 3, 4]),       # R < 0 (wide edges)
    ("Flat-4",  [4, 4, 4]),             # R = 0, different shape
    ("Lens",    [2, 4, 4, 4, 2]),       # R > 0, different profile
    ("Hourglass", [4, 2, 2, 2, 4]),     # R < 0, pinched middle
]

all_results = {}
for name, widths in configs:
    n_cc, curvature, log_Z = grid_stats(widths, name)
    all_results[name] = (sum(widths), n_cc, curvature, log_Z)

# THE KEY TEST: is log Z a function of curvature?
print(f"\n{'='*50}")
print(f"THE KEY TEST: log Z vs curvature")
print(f"{'='*50}")

for beta in [1.0, 2.0]:
    print(f"\nβ = {beta}:")
    print(f"{'Config':>12} {'n':>4} {'R':>4} {'log Z':>10} {'log Z/n':>10}")
    for name, widths in configs:
        n, n_cc, R, log_Z = all_results[name]
        print(f"{name:>12} {n:>4} {R:>4} {log_Z[beta]:>10.4f} {log_Z[beta]/n:>10.4f}")
    
    # For configs with same n: is Δlog Z proportional to ΔR?
    # Compare Flat vs Sphere vs Saddle (all n=15)
    flat_Z = all_results["Flat"][3][beta]
    sphere_Z = all_results["Sphere"][3][beta]
    saddle_Z = all_results["Saddle"][3][beta]
    flat_R = all_results["Flat"][2]
    sphere_R = all_results["Sphere"][2]
    saddle_R = all_results["Saddle"][2]
    
    print(f"\n  Same-size comparison (n=15):")
    print(f"  Δlog Z (Sphere - Flat) = {sphere_Z - flat_Z:.4f}, ΔR = {sphere_R - flat_R}")
    print(f"  Δlog Z (Saddle - Flat) = {saddle_Z - flat_Z:.4f}, ΔR = {saddle_R - flat_R}")
    if (sphere_R - flat_R) != 0 and (saddle_R - flat_R) != 0:
        ratio1 = (sphere_Z - flat_Z) / (sphere_R - flat_R)
        ratio2 = (saddle_Z - flat_Z) / (saddle_R - flat_R)
        print(f"  Δlog Z / ΔR (Sphere): {ratio1:.4f}")
        print(f"  Δlog Z / ΔR (Saddle): {ratio2:.4f}")
        print(f"  MATCH: {'YES' if abs(ratio1 - ratio2) < 0.1*abs(ratio1) else 'NO'}")
        print(f"  (If these ratios are equal, log Z = α·n + β·R + ... = discrete EH action)")

print(f"\n{'='*50}")
print("INTERPRETATION")
print(f"{'='*50}")
print("""
If Δlog Z / ΔR is the SAME for Sphere and Saddle,
then log Z = (area term) + (curvature coefficient) · R + higher order.

This would mean: the BD partition function on convex subsets IS
the discrete Einstein-Hilbert action, with the curvature coefficient
determined by the combinatorics — not put in by hand.
""")
