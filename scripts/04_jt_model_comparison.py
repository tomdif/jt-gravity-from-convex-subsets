"""
Test: does log Z = a·R + b·∫(w'/w)² + c·(boundary terms) + d explain the data?

The JT gravity action on a 2D surface with dilaton Φ:
  S_JT = ∫ Φ·R √g + boundary terms

For a variable-width grid with profile w(t):
  - The conformal factor is related to w(t)
  - The Schwarzian derivative ~ (w''/w - (w'/w)²/2)
  - The JT boundary action ~ ∫ (w'/w)² dt

Geometric invariants of a width profile w(0),...,w(T-1):
  R  = Σ (w_{i-1} + w_{i+1} - 2w_i)  [discrete Laplacian = curvature]
  S  = Σ ((w_{i+1} - w_i)/w_i)²       [discrete Schwarzian-like]
  B  = w(0) + w(T-1)                   [boundary widths]
  V  = Σ w_i                           [volume = n, fixed]
  W  = min(w_i)                        [bottleneck width]
  L  = Σ |w_{i+1} - w_i|              [total variation]
  E  = Σ log(w_i)                      [log-volume / entropy potential]
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

def compute_invariants(widths):
    T = len(widths)
    w = np.array(widths, dtype=float)
    
    # Curvature
    R = sum(widths[i-1] + widths[i+1] - 2*widths[i] for i in range(1, T-1))
    
    # Schwarzian-like: Σ ((w_{i+1} - w_i)/w_i)²
    schwarzian = sum(((w[i+1] - w[i]) / w[i])**2 for i in range(T-1))
    
    # Log-Schwarzian: Σ (log w_{i+1} - log w_i)²
    log_schwarz = sum((math.log(w[i+1]) - math.log(w[i]))**2 for i in range(T-1))
    
    # Boundary
    boundary = w[0] + w[-1]
    
    # Bottleneck
    bottleneck = min(widths)
    
    # Total variation
    tv = sum(abs(w[i+1] - w[i]) for i in range(T-1))
    
    # Log-volume (sum of log widths)
    log_vol = sum(math.log(wi) for wi in widths)
    
    # Second derivative energy: Σ (w'' / w)²
    curv_energy = sum(((w[i-1] + w[i+1] - 2*w[i]) / w[i])**2 for i in range(1, T-1))
    
    # Max width
    max_w = max(widths)
    
    return {
        'R': R, 'schwarzian': schwarzian, 'log_schwarz': log_schwarz,
        'boundary': boundary, 'bottleneck': bottleneck, 'tv': tv,
        'log_vol': log_vol, 'curv_energy': curv_energy, 'max_w': max_w
    }

# Generate all n=15, 5-row profiles
profiles = set()
for w1 in range(1, 8):
    for w2 in range(1, 8):
        for w3 in range(1, 8):
            for w4 in range(1, 8):
                w5 = 15 - w1 - w2 - w3 - w4
                if 1 <= w5 <= 7:
                    profiles.add((w1, w2, w3, w4, w5))

# Select diverse subset
by_R = defaultdict(list)
for p in profiles:
    R = sum(p[i-1] + p[i+1] - 2*p[i] for i in range(1, 4))
    by_R[R].append(p)

selected = []
for R in sorted(by_R.keys()):
    profs = sorted(by_R[R])
    selected.append(profs[0])
    if len(profs) > 1:
        selected.append(profs[len(profs)//2])

print(f"Computing log Z for {len(selected)} profiles...")

beta = 1.0
data = []
for i, widths in enumerate(selected):
    inv = compute_invariants(widths)
    sbd_list = enumerate_cc_varwidth(widths)
    sbd_arr = np.array(sbd_list, dtype=float)
    w = np.exp(-beta * sbd_arr)
    logZ = math.log(w.sum())
    data.append((widths, inv, logZ))
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(selected)} done")

print(f"All done.\n")

# Extract arrays
logZ_arr = np.array([d[2] for d in data])
R_arr = np.array([d[1]['R'] for d in data])
S_arr = np.array([d[1]['schwarzian'] for d in data])
LS_arr = np.array([d[1]['log_schwarz'] for d in data])
B_arr = np.array([d[1]['boundary'] for d in data])
W_arr = np.array([d[1]['bottleneck'] for d in data])
TV_arr = np.array([d[1]['tv'] for d in data])
LV_arr = np.array([d[1]['log_vol'] for d in data])
CE_arr = np.array([d[1]['curv_energy'] for d in data])
MW_arr = np.array([d[1]['max_w'] for d in data])

n_pts = len(data)

def fit_and_report(name, X, y):
    A = np.column_stack([X, np.ones(len(y))])
    coeffs, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ coeffs
    residuals = y - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    rms = np.sqrt(np.mean(residuals**2))
    
    terms = " + ".join(f"{coeffs[i]:+.4f}·{name[i]}" for i in range(len(name)))
    terms += f" + {coeffs[-1]:+.4f}"
    print(f"  {terms}")
    print(f"  R² = {r2:.4f}, RMS = {rms:.4f}")
    return r2, rms, coeffs

print("="*60)
print(f"MODEL COMPARISON: {n_pts} data points, β = {beta}")
print("="*60)

# Model 1: R only
print(f"\n1. log Z = a·R + const")
r2_1, _, _ = fit_and_report(['R'], np.column_stack([R_arr]), logZ_arr)

# Model 2: R + Schwarzian
print(f"\n2. log Z = a·R + b·Schwarzian + const")
r2_2, _, _ = fit_and_report(['R', 'Schwarz'], np.column_stack([R_arr, S_arr]), logZ_arr)

# Model 3: R + log-Schwarzian
print(f"\n3. log Z = a·R + b·LogSchwarz + const")
r2_3, _, _ = fit_and_report(['R', 'LogSch'], np.column_stack([R_arr, LS_arr]), logZ_arr)

# Model 4: R + log-volume (sum log w_i)
print(f"\n4. log Z = a·R + b·Σlog(w_i) + const")
r2_4, _, _ = fit_and_report(['R', 'LogVol'], np.column_stack([R_arr, LV_arr]), logZ_arr)

# Model 5: R + bottleneck
print(f"\n5. log Z = a·R + b·min(w) + const")
r2_5, _, _ = fit_and_report(['R', 'MinW'], np.column_stack([R_arr, W_arr]), logZ_arr)

# Model 6: log-volume alone
print(f"\n6. log Z = a·Σlog(w_i) + const")
r2_6, _, _ = fit_and_report(['LogVol'], np.column_stack([LV_arr]), logZ_arr)

# Model 7: R + log-Schwarzian + boundary
print(f"\n7. log Z = a·R + b·LogSchwarz + c·boundary + const")
r2_7, _, _ = fit_and_report(['R', 'LogSch', 'Bdy'],
    np.column_stack([R_arr, LS_arr, B_arr]), logZ_arr)

# Model 8: log-volume + log-Schwarzian
print(f"\n8. log Z = a·Σlog(w) + b·LogSchwarz + const")
r2_8, _, _ = fit_and_report(['LogVol', 'LogSch'],
    np.column_stack([LV_arr, LS_arr]), logZ_arr)

# Model 9: R + log-volume + log-Schwarzian
print(f"\n9. log Z = a·R + b·Σlog(w) + c·LogSchwarz + const")
r2_9, _, _ = fit_and_report(['R', 'LogVol', 'LogSch'],
    np.column_stack([R_arr, LV_arr, LS_arr]), logZ_arr)

# Model 10: KITCHEN SINK
print(f"\n10. log Z = a·R + b·LogVol + c·LogSch + d·Bdy + e·MinW + const")
r2_10, _, _ = fit_and_report(['R', 'LogVol', 'LogSch', 'Bdy', 'MinW'],
    np.column_stack([R_arr, LV_arr, LS_arr, B_arr, W_arr]), logZ_arr)

print(f"\n{'='*60}")
print(f"SUMMARY: R² comparison")
print(f"{'='*60}")
models = [
    ("R only", r2_1, 1),
    ("R + Schwarzian", r2_2, 2),
    ("R + LogSchwarzian", r2_3, 2),
    ("R + LogVolume", r2_4, 2),
    ("R + MinWidth", r2_5, 2),
    ("LogVolume only", r2_6, 1),
    ("R + LogSch + Bdy", r2_7, 3),
    ("LogVol + LogSch", r2_8, 2),
    ("R + LogVol + LogSch", r2_9, 3),
    ("Kitchen sink", r2_10, 5),
]
for name, r2, nparams in sorted(models, key=lambda x: -x[1]):
    bar = "█" * int(r2 * 40)
    print(f"  {name:>22}: R²={r2:.3f} ({nparams}p) {bar}")
