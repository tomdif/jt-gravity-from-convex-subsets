"""
BD-weighted partition function and 2-point correlator on [2]^4.
|CC([2]^4)| = 3938 convex subsets, 16 elements.

d=4 spacetime: the graviton propagator should go as 1/r^{d-2} = 1/r^2.
"""
import numpy as np
from collections import defaultdict
import math, time

M = 2
D = 4

# Elements of [2]^4
cells = []
for i in range(M):
    for j in range(M):
        for k in range(M):
            for l in range(M):
                cells.append((i,j,k,l))
n = len(cells)
cell_idx = {c: i for i, c in enumerate(cells)}

print(f"Grid [{M}]^{D}: {n} elements")

def le_prod(a, b):
    return all(a[i] <= b[i] for i in range(D))

def is_convex(S_set):
    for a in S_set:
        for b in S_set:
            if le_prod(a, b):
                for c in cells:
                    if le_prod(a, c) and le_prod(c, b) and c not in S_set:
                        return False
    return True

def sbd(S_set):
    """S_BD = |S| - links. Links: covering relations in the product order."""
    links = 0
    for a in S_set:
        for d_idx in range(D):
            b = list(a)
            b[d_idx] += 1
            b = tuple(b)
            if b in S_set:
                links += 1
    return len(S_set) - links

# Enumerate all convex subsets of [2]^4
t0 = time.time()
all_subsets = []
for bits in range(1 << n):
    S = frozenset(cells[k] for k in range(n) if bits & (1 << k))
    if is_convex(S):
        s = sbd(S)
        all_subsets.append((s, S))

elapsed = time.time() - t0
print(f"|CC([{M}]^{D})| = {len(all_subsets)} ({elapsed:.1f}s)")

# S_BD distribution
sbd_dist = defaultdict(int)
for s, S in all_subsets:
    sbd_dist[s] += 1

print(f"\nS_BD distribution:")
for e in sorted(sbd_dist.keys()):
    print(f"  S_BD = {e:3d}: {sbd_dist[e]:5d} subsets")

# Full grid S_BD
full = frozenset(cells)
print(f"\nS_BD(full grid) = {sbd(full)}")
print(f"Expected: n - d*n/2 = {n} - {D}*{n//2} = {n - D*(M-1)*M**(D-1)}")

# Partition function
sbd_arr = np.array([s[0] for s in all_subsets], dtype=float)
sizes = np.array([len(s[1]) for s in all_subsets], dtype=float)

print(f"\nPartition function Z(β):")
print(f"{'β':>6} {'Z':>14} {'<S_BD>':>10} {'<|S|>':>8} {'Cv':>8}")
for beta in [0, 0.5, 1.0, 2.0, 5.0, -0.5, -1.0]:
    w = np.exp(-beta * sbd_arr)
    Z = w.sum()
    avg_s = (sbd_arr * w).sum() / Z
    avg_n = (sizes * w).sum() / Z
    var_s = (sbd_arr**2 * w).sum() / Z - avg_s**2
    Cv = beta**2 * var_s
    print(f"{beta:>6.1f} {Z:>14.2f} {avg_s:>10.4f} {avg_n:>8.3f} {Cv:>8.4f}")

# 2-point correlator at β = 1.0
beta = 1.0
w = np.exp(-beta * sbd_arr)
Z = w.sum()

phi_avg = np.zeros(n)
phi2 = np.zeros((n, n))

for idx, (s, S) in enumerate(all_subsets):
    for c in S:
        ci = cell_idx[c]
        phi_avg[ci] += w[idx]
        for c2 in S:
            phi2[ci, cell_idx[c2]] += w[idx]

phi_avg /= Z
phi2 /= Z
G = phi2 - np.outer(phi_avg, phi_avg)

print(f"\n<φ(x)> at β={beta}:")
for i, c in enumerate(cells):
    print(f"  {c}: {phi_avg[i]:.4f}")

# Manhattan distance correlator
def manhattan(a, b):
    return sum(abs(a[i]-b[i]) for i in range(D))

print(f"\nConnected correlator G(r) averaged over Manhattan distance:")
dist_corr = defaultdict(list)
for i in range(n):
    for j in range(i, n):
        d = manhattan(cells[i], cells[j])
        dist_corr[d].append(G[i,j])

for d in sorted(dist_corr.keys()):
    vals = dist_corr[d]
    avg = sum(vals)/len(vals)
    print(f"  r={d}: G_avg = {avg:>10.6f} ({len(vals)} pairs)")

# Power law fit for r >= 1
r_vals = []
G_vals = []
for d in sorted(dist_corr.keys()):
    if d >= 1:
        r_vals.append(d)
        G_vals.append(sum(dist_corr[d])/len(dist_corr[d]))

r_arr = np.array(r_vals, dtype=float)
G_arr = np.array(G_vals, dtype=float)
positive = G_arr > 0
if positive.sum() >= 2:
    log_r = np.log(r_arr[positive])
    log_G = np.log(G_arr[positive])
    slope, intercept = np.polyfit(log_r, log_G, 1)
    print(f"\nPower law fit G(r) ~ {math.exp(intercept):.4f} / r^{-slope:.3f}")
    print(f"Exponent α = {-slope:.3f}")
    print(f"d=4 graviton prediction: α = d-2 = 2")
    print(f"Match: {'YES' if abs(-slope - 2) < 0.5 else 'NO'} (α = {-slope:.2f} vs predicted 2)")

# Eigenvalue analysis of the correlation matrix
print(f"\nEigenvalues of the {n}×{n} correlation matrix G:")
eigenvalues = np.linalg.eigvalsh(G)
for i, ev in enumerate(sorted(eigenvalues, reverse=True)):
    if abs(ev) > 1e-10:
        print(f"  λ_{i} = {ev:>10.6f}")
    if i > 6: break

# Count near-degenerate eigenvalues (graviton polarizations)
ev_sorted = sorted(eigenvalues, reverse=True)
if len(ev_sorted) >= 2:
    print(f"\nTop eigenvalue ratio λ₁/λ₂ = {ev_sorted[0]/ev_sorted[1]:.3f}")
    print(f"(For 2 graviton polarizations, expect ratio ≈ 1)")
    
    # Count eigenvalues within 10% of the largest
    threshold = ev_sorted[0] * 0.9
    n_near = sum(1 for ev in ev_sorted if ev > threshold)
    print(f"Eigenvalues within 90% of max: {n_near}")
    print(f"(d=4 graviton: expect 2 dominant modes with equal eigenvalue)")

print(f"\n{'='*60}")
print("SUMMARY FOR d=4 SPACETIME")
print(f"{'='*60}")
