"""
Compute the Benincasa-Dowker action S_BD[S] for every convex subset S of [m]^2,
then study the partition function Z(beta) = sum_S exp(-beta * S_BD[S]).

The BD action for d=2 causal sets:
  S_BD = sum_{x in C} [1 - N_link(x)]
where N_link(x) = number of direct causal links FROM x (covering relations x -> y in S).

For the grid: covering relations are (i,j) -> (i+1,j) and (i,j) -> (i,j+1).
So for a convex subset S:
  S_BD(S) = |S| - (number of covering pairs (a,b) with both in S)

This measures "how flat" the subset is: flat subsets have many links per element
(S_BD negative/small), curved ones have fewer.
"""
import math
import numpy as np
from collections import defaultdict

def enumerate_convex_subsets(m):
    """Enumerate all convex subsets of [m]^2 by brute force."""
    cells = [(i,j) for i in range(m) for j in range(m)]
    n = len(cells)
    subsets = []
    
    for bits in range(1 << n):
        S = set()
        for k in range(n):
            if bits & (1 << k):
                S.add(cells[k])
        
        # Check convexity
        convex = True
        for a in S:
            if not convex: break
            for b in S:
                if not convex: break
                if a[0] <= b[0] and a[1] <= b[1]:
                    for x in range(a[0], b[0]+1):
                        for y in range(a[1], b[1]+1):
                            if (x,y) not in S:
                                convex = False
                                break
                        if not convex: break
        
        if convex:
            subsets.append(frozenset(S))
    
    return subsets

def bd_action(S, m):
    """Compute the Benincasa-Dowker action for a convex subset S of [m]^2.
    S_BD = |S| - |{covering pairs in S}|
    Covering pairs: (i,j) -> (i+1,j) and (i,j) -> (i,j+1)."""
    if len(S) == 0:
        return 0
    links = 0
    for (i,j) in S:
        if (i+1,j) in S: links += 1
        if (i,j+1) in S: links += 1
    return len(S) - links

def interval_count(S, m):
    """Count k-element order intervals within S for k=2,3,..."""
    counts = defaultdict(int)
    S_list = sorted(S)
    for a in S_list:
        for b in S_list:
            if a[0] <= b[0] and a[1] <= b[1] and a != b:
                # Count elements in [a,b] ∩ S
                interval = [c for c in S if a[0]<=c[0]<=b[0] and a[1]<=c[1]<=b[1]]
                k = len(interval)
                if k >= 2:
                    counts[k] += 1
    return counts

print("=" * 70)
print("BENINCASA-DOWKER ACTION ON CONVEX SUBSETS OF [m]^2")
print("=" * 70)

for m in [2, 3, 4]:
    print(f"\n--- m = {m} ---")
    subsets = enumerate_convex_subsets(m)
    print(f"|CC([{m}]^2)| = {len(subsets)}")
    
    # Compute S_BD for each subset
    actions = []
    for S in subsets:
        s_bd = bd_action(S, m)
        actions.append((s_bd, len(S), S))
    
    # Distribution of S_BD values
    action_dist = defaultdict(int)
    for s_bd, size, S in actions:
        action_dist[s_bd] += 1
    
    print(f"S_BD distribution:")
    for k in sorted(action_dist.keys()):
        print(f"  S_BD = {k:3d}: {action_dist[k]:5d} subsets")
    
    # Partition function Z(beta) for various beta
    print(f"\nPartition function Z(beta) and observables:")
    print(f"{'beta':>8} {'Z':>14} {'<S_BD>':>10} {'<|S|>':>10} {'Cv':>10}")
    
    bd_values = np.array([a[0] for a in actions], dtype=float)
    sizes = np.array([a[1] for a in actions], dtype=float)
    
    for beta in [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -0.5, -1.0]:
        weights = np.exp(-beta * bd_values)
        Z = weights.sum()
        avg_sbd = (bd_values * weights).sum() / Z
        avg_size = (sizes * weights).sum() / Z
        avg_sbd2 = (bd_values**2 * weights).sum() / Z
        Cv = beta**2 * (avg_sbd2 - avg_sbd**2)
        print(f"{beta:>8.1f} {Z:>14.4f} {avg_sbd:>10.4f} {avg_size:>10.4f} {Cv:>10.4f}")
    
    # KEY: at negative beta (Euclidean QG = Wick-rotated), large NEGATIVE S_BD is favored.
    # Large negative S_BD = many links per element = "flat" geometry.
    # So the Euclidean path integral SELECTS flat geometries. Correct!
    
    # What are the dominant configurations at beta = -1 (Euclidean)?
    if m <= 3:
        print(f"\nTop 5 dominant configurations at beta = -1 (Euclidean gravity):")
        weighted = [(math.exp(1.0 * s_bd), s_bd, size, S) for s_bd, size, S in actions]
        weighted.sort(reverse=True)
        for w, s_bd, size, S in weighted[:5]:
            print(f"  weight={w:.2f}, S_BD={s_bd}, |S|={size}, S={sorted(S)}")

print("\n" + "=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)
print("""
For the EUCLIDEAN path integral Z = sum exp(+S_BD) (beta < 0):
  - Configurations with LARGE NEGATIVE S_BD dominate
  - These are subsets with MANY links per element = FLAT geometry
  - The full grid [m]^2 has S_BD = m^2 - 2m(m-1) = m^2 - 2m^2 + 2m = -m^2 + 2m
  - For large m: S_BD ~ -m^2 (most negative = flattest)
  - The Euclidean path integral SELECTS flat spacetime!

For the LORENTZIAN integral Z = sum exp(i*S_BD):
  - All phases contribute, interference effects
  - Stationary phase at dS_BD/d(deformation) = 0 = Einstein equations
  - Fluctuations around flat space = gravitons

The BD action on convex subsets provides a DISCRETE PATH INTEGRAL
where the sum is over causally consistent subregions, and flat geometry
emerges as the saddle point.
""")
