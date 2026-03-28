"""
n=30 JT regression: 2D variable-width grids with Σw_t = 30.
Use the CORRECT DP (not brute force) for exact CC enumeration.
Then compute Z(β) and fit R + LogSch.
"""
import numpy as np
import math, time
from collections import defaultdict

def count_cc_and_sbd_dp(row_widths):
    """Count convex subsets and compute S_BD distribution using the
    row-by-row DP with gap handling (the correct v4 algorithm).
    Returns dict: S_BD_value -> count."""
    m_rows = len(row_widths)
    max_w = max(row_widths)
    
    # State: (phase, last_lo, last_hi, min_lo)
    # Phase 0: not_started, 1: contiguous, 2: gap
    # For each state, track S_BD value distribution
    # 
    # Actually tracking full S_BD distribution per state is expensive.
    # Instead: compute Z(β) directly for several β values.
    # Z(β) = Σ_S exp(-β * S_BD(S))
    #
    # S_BD(S) = |S| - links(S)
    # When we add a row with interval [lo, hi], we add (hi-lo+1) elements
    # and (hi-lo) horizontal links. Vertical links to the previous row:
    # if prev row had [lo_p, hi_p], vertical links = max(0, min(hi, hi_p) - max(lo, lo_p) + 1)
    # when the intervals overlap.
    
    # For efficiency: track (state, accumulated_weight) where weight = exp(-β * S_BD_so_far)
    # at multiple β values simultaneously.
    
    betas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    n_beta = len(betas)
    beta_arr = np.array(betas)
    
    # State: (last_lo, last_hi, min_lo, phase)
    # Weight: array of n_beta values
    
    # Phase 0: not started
    # Phase 1: contiguous (last_lo, last_hi, min_lo)
    # Phase 2: gap (min_lo)
    
    # Initialize: not_started with weight = 1 for all β (empty set has S_BD=0)
    Z = np.ones(n_beta)  # accumulate final weights
    
    # DP states: dict from state_key -> weight_array
    # not_started
    dp_not_started = np.ones(n_beta)
    # contiguous: (last_lo, last_hi, min_lo) -> weight
    dp_cont = {}
    # gap: min_lo -> weight
    dp_gap = {}
    
    for row_idx in range(m_rows):
        w = row_widths[row_idx]
        new_cont = {}
        new_gap = {}
        
        # not_started + add interval [lo, hi]
        for lo in range(w):
            for hi in range(lo, w):
                n_elem = hi - lo + 1
                h_links = hi - lo  # horizontal links
                delta_sbd = n_elem - h_links  # = 1 always
                weight = dp_not_started * np.exp(-beta_arr * delta_sbd)
                key = (lo, hi, lo)
                if key in new_cont:
                    new_cont[key] += weight
                else:
                    new_cont[key] = weight.copy()
        
        # contiguous + skip -> gap
        for (lo, hi, mlo), weight in dp_cont.items():
            if mlo in dp_gap:
                dp_gap[mlo] = dp_gap.get(mlo, np.zeros(n_beta))
            # Actually accumulate into new_gap for THIS row's gap
            # But gap means we skip this row, so the state persists
            pass
        # Actually: skip means this row is empty.
        # contiguous -> gap: all contiguous states become gap states
        for (lo, hi, mlo), weight in dp_cont.items():
            if mlo in new_gap:
                new_gap[mlo] += weight
            else:
                new_gap[mlo] = weight.copy()
        
        # contiguous + add interval [lo_new, hi_new]
        for (lo_p, hi_p, mlo), weight in dp_cont.items():
            for lo in range(w):
                for hi in range(lo, w):
                    # Check transition constraint
                    if lo_p <= hi:
                        if not (lo <= lo_p and hi_p >= hi):
                            continue
                    # else: lo_p > hi, no constraint
                    
                    # Compute delta S_BD
                    n_elem = hi - lo + 1
                    h_links = hi - lo
                    # Vertical links: overlap of [lo, hi] and [lo_p, hi_p]
                    v_lo = max(lo, lo_p)
                    v_hi = min(hi, hi_p)
                    v_links = max(0, v_hi - v_lo + 1)
                    delta_sbd = n_elem - h_links - v_links  # = 1 - v_links
                    
                    new_weight = weight * np.exp(-beta_arr * delta_sbd)
                    new_mlo = min(mlo, lo)
                    key = (lo, hi, new_mlo)
                    if key in new_cont:
                        new_cont[key] += new_weight
                    else:
                        new_cont[key] = new_weight.copy()
        
        # gap + skip -> gap (stays)
        for mlo, weight in dp_gap.items():
            if mlo in new_gap:
                new_gap[mlo] += weight
            else:
                new_gap[mlo] = weight.copy()
        
        # gap + add interval [lo, hi] -> contiguous (if gap-compatible)
        for mlo, weight in dp_gap.items():
            for lo in range(w):
                for hi in range(lo, w):
                    if hi < mlo:
                        n_elem = hi - lo + 1
                        h_links = hi - lo
                        delta_sbd = n_elem - h_links  # = 1 (no vertical links after gap)
                        new_weight = weight * np.exp(-beta_arr * delta_sbd)
                        new_mlo = min(mlo, lo)
                        key = (lo, hi, new_mlo)
                        if key in new_cont:
                            new_cont[key] += new_weight
                        else:
                            new_cont[key] = new_weight.copy()
        
        # not_started stays (skip this row)
        # dp_not_started unchanged
        
        dp_cont = new_cont
        dp_gap = new_gap
    
    # Final: Z = not_started + all contiguous + all gap
    Z = dp_not_started.copy()  # empty set
    for weight in dp_cont.values():
        Z += weight
    for weight in dp_gap.values():
        Z += weight
    
    return dict(zip(betas, Z))

def compute_R(widths):
    return sum(widths[i-1]+widths[i+1]-2*widths[i] for i in range(1, len(widths)-1))

def compute_log_sch(widths):
    w = np.array(widths, dtype=float)
    return sum((math.log(w[i+1])-math.log(w[i]))**2 for i in range(len(w)-1))

# Verify on known values first
print("VERIFICATION on small grids:")
for widths, expected_cc in [([2,2], 13), ([3,3,3], 114), ([3,3,3,3,3], 781)]:
    Z = count_cc_and_sbd_dp(widths)
    print(f"  {widths}: |CC| = {Z[0.0]:.0f} (expected {expected_cc}) {'✓' if abs(Z[0.0]-expected_cc)<0.5 else '✗'}")

# Generate n=30 profiles with 5 rows
print(f"\nGenerating n=30 profiles (5 rows, Σw=30)...")
profiles_30 = set()
for w1 in range(1, 15):
    for w2 in range(1, 15):
        for w3 in range(1, 15):
            for w4 in range(1, 15):
                w5 = 30 - w1 - w2 - w3 - w4
                if 1 <= w5 <= 14:
                    profiles_30.add((w1, w2, w3, w4, w5))

by_R = defaultdict(list)
for p in profiles_30:
    by_R[compute_R(p)].append(p)

# Select ~50 diverse profiles
selected = []
for R in sorted(by_R.keys()):
    profs = sorted(by_R[R])
    selected.append(profs[0])
    if len(profs) > 2:
        selected.append(profs[len(profs)//2])

# Limit to ~50
if len(selected) > 55:
    selected = selected[::len(selected)//50 + 1][:50]

print(f"Selected {len(selected)} profiles")
print(f"Computing Z(β) for each...")

t0 = time.time()
data = []
for i, widths in enumerate(selected):
    R = compute_R(widths)
    LS = compute_log_sch(widths)
    Z = count_cc_and_sbd_dp(widths)
    data.append({'widths': widths, 'R': R, 'log_sch': LS, 'Z': Z, 'n': sum(widths)})
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(selected)} ({time.time()-t0:.1f}s)")

print(f"All done in {time.time()-t0:.1f}s")

# Regression at each β
R_arr = np.array([d['R'] for d in data])
LS_arr = np.array([d['log_sch'] for d in data])
n_pts = len(data)

print(f"\n{'='*60}")
print(f"JT REGRESSION: n=30, {n_pts} profiles")
print(f"{'='*60}")

print(f"\n{'β':>6} {'R²(R)':>8} {'R²(R+LS)':>10} {'a(R)':>10} {'b(LS)':>10}")
print("-"*50)

for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
    logZ = np.array([math.log(d['Z'][beta]) for d in data])
    
    # R only
    A1 = np.column_stack([R_arr, np.ones(n_pts)])
    c1 = np.linalg.lstsq(A1, logZ, rcond=None)[0]
    r2_1 = 1 - np.var(logZ - A1@c1)/np.var(logZ)
    
    # R + LogSch
    A2 = np.column_stack([R_arr, LS_arr, np.ones(n_pts)])
    c2 = np.linalg.lstsq(A2, logZ, rcond=None)[0]
    r2_2 = 1 - np.var(logZ - A2@c2)/np.var(logZ)
    
    print(f"{beta:>6.1f} {r2_1:>8.4f} {r2_2:>10.4f} {c2[0]:>10.4f} {c2[1]:>10.4f}")

print(f"\nFor comparison, n=15 gave: R²(R)=0.269, R²(R+LS)=0.846")
