"""
RNA Convex Subset Partition Function.

Z(β) = Σ_{S ∈ CC(P)} exp(-β · E(S))

where P is the base-pair containment poset and E(S) is the free energy
of the sub-structure S, estimated from nearest-neighbor stacking parameters.

For each convex subset S of base pairs, E(S) = sum of stacking energies
for consecutive pairs in S that form a stack.

Nearest-neighbor stacking energies (kcal/mol, Turner 2004 simplified):
  AU/AU stack: -0.9
  GC/GC stack: -3.4
  GU/GU stack: -0.5
  AU/GC: -2.2
  Mixed: -1.3 (average)

Since we don't have the actual sequences for most Rfam families,
use a SIMPLIFIED energy: E(S) = -1.5 * (number of stacking pairs in S)
where a "stacking pair" is two base pairs (i,j) and (i+1,j-1) that are
consecutive in the helix.

Then compute:
1. Z(β) at several β values
2. Per-element occupation probability <φ(x)> = P(pair x is in S)
3. Check: do terminal pairs have lower <φ> than interior pairs?
4. Identify "load-bearing" vs "fragile" structural elements
"""
import numpy as np
import math
from collections import defaultdict

def parse_pairs(db):
    stack = []; pairs = []
    for i, c in enumerate(db):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack: pairs.append((stack.pop(), i))
    return sorted(pairs)

def find_stacks(pairs):
    """Find stacking pairs: (i,j) and (i+1,j-1) both in the pair list."""
    pair_set = set(pairs)
    stacks = []
    for (i, j) in pairs:
        if (i+1, j-1) in pair_set:
            stacks.append(((i,j), (i+1,j-1)))
    return stacks

def energy_of_subset(S_indices, pairs, stacks):
    """Energy of a convex subset of base pairs.
    E = -1.5 * (number of stacking interactions present in S)."""
    S_set = set(S_indices)
    e = 0.0
    for (p1_idx, p2_idx) in stacks:
        # Find indices of these pairs in the pairs list
        i1 = pairs.index(p1_idx) if p1_idx in pairs else -1
        i2 = pairs.index(p2_idx) if p2_idx in pairs else -1
        # Actually we need to map pair tuples to indices
        pass
    
    # Simpler: precompute which pair indices form stacks
    return e

def compute_rna_Z(db_string, name=""):
    """Compute the convex subset partition function for an RNA structure."""
    pairs = parse_pairs(db_string)
    m = len(pairs)
    if m == 0 or m > 22:
        return None
    
    # Build containment poset
    le = [[False]*m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            ia, ja = pairs[a]; ib, jb = pairs[b]
            le[a][b] = (ib <= ia and ja <= jb)
    
    # Find stacking pairs (consecutive in helix)
    pair_set = {p: idx for idx, p in enumerate(pairs)}
    stack_pairs = []  # list of (idx_a, idx_b) where pairs[a] stacks on pairs[b]
    for idx_a, (ia, ja) in enumerate(pairs):
        partner = (ia+1, ja-1)
        if partner in pair_set:
            idx_b = pair_set[partner]
            stack_pairs.append((idx_a, idx_b))
    
    # Enumerate all convex subsets and compute energy
    betas = [0, 0.5, 1.0, 2.0, 5.0]
    n_subsets = 1 << m
    
    # For each subset: check convexity, compute energy
    results = []
    for bits in range(n_subsets):
        S = [k for k in range(m) if bits & (1 << k)]
        
        # Check convexity
        convex = True
        for a in S:
            if not convex: break
            for b in S:
                if not convex: break
                if le[a][b]:
                    for c in range(m):
                        if le[a][c] and le[c][b] and not (bits & (1 << c)):
                            convex = False; break
        
        if not convex: continue
        
        # Energy: -1.5 per stacking pair present in S
        S_set = set(S)
        n_stacks = sum(1 for (a, b) in stack_pairs if a in S_set and b in S_set)
        energy = -1.5 * n_stacks
        
        results.append((bits, S, energy))
    
    # Compute Z(β) and per-element occupation
    print(f"\n{'='*60}")
    print(f"{name}: {m} base pairs, {len(results)} convex subsets, {len(stack_pairs)} stack pairs")
    print(f"{'='*60}")
    
    for beta in betas:
        weights = np.array([math.exp(-beta * e) for _, _, e in results])
        Z = weights.sum()
        
        # Per-pair occupation probability
        phi = np.zeros(m)
        for idx, (bits, S, e) in enumerate(results):
            for k in S:
                phi[k] += weights[idx]
        phi /= Z
        
        if beta == 1.0:
            # Classify each pair
            print(f"\nβ = {beta}: Z = {Z:.2f}, log Z = {math.log(Z):.2f}")
            print(f"  Per-pair occupation ⟨φ(pair)⟩:")
            
            # Determine if each pair is terminal, internal, or junction
            for idx in range(m):
                i, j = pairs[idx]
                # Is this pair a terminal pair? (no pair nested immediately inside)
                has_inner = any((i+1, j-1) == pairs[k] for k in range(m))
                has_outer = any((i-1, j+1) == pairs[k] for k in range(m))
                
                if not has_inner and not has_outer:
                    pos = "isolated"
                elif not has_inner:
                    pos = "TERMINAL"
                elif not has_outer:
                    pos = "outermost"
                else:
                    pos = "interior"
                
                print(f"    pair ({i:>3},{j:>3}): ⟨φ⟩ = {phi[idx]:.4f}  [{pos}]")
            
            # Summary
            terminal_phi = [phi[idx] for idx in range(m) 
                          if not any((pairs[idx][0]+1, pairs[idx][1]-1) == pairs[k] for k in range(m))]
            interior_phi = [phi[idx] for idx in range(m)
                          if any((pairs[idx][0]+1, pairs[idx][1]-1) == pairs[k] for k in range(m))
                          and any((pairs[idx][0]-1, pairs[idx][1]+1) == pairs[k] for k in range(m))]
            
            if terminal_phi and interior_phi:
                print(f"\n  TERMINAL pairs: mean ⟨φ⟩ = {np.mean(terminal_phi):.4f}")
                print(f"  INTERIOR pairs: mean ⟨φ⟩ = {np.mean(interior_phi):.4f}")
                if np.mean(interior_phi) > np.mean(terminal_phi):
                    print(f"  → Interior > Terminal: BOUNDARY DOMINANCE ✓")
                    print(f"    (RNA analogue of 'corners fluctuate, bulk doesn't')")
                else:
                    print(f"  → No boundary dominance pattern")
    
    # β-dependence
    print(f"\n  β-scaling of log Z:")
    for beta in betas:
        if beta == 0:
            lZ = math.log(len(results))
        else:
            weights = np.array([math.exp(-beta * e) for _, _, e in results])
            lZ = math.log(weights.sum())
        print(f"    β={beta:.1f}: log Z = {lZ:.3f}")
    
    return results

# Test structures
structures = [
    ("Hairpin",       "(((...)))"),
    ("Simple stem",   "((...))"),
    ("Two hairpins",  "(((...)))...(((...)))"),
    ("Nested",        "(((((...)))))"),
    ("Internal loop", "(((..((...))..)))"),
    ("Multi-branch",  "(((...)(...)(...)))"),
    ("tRNA-like",     "(((..(((...)))..(((...)))...)))"),
    ("Complex",       "(((..(((...)))..((...))..)))"),
]

print("RNA CONVEX SUBSET PARTITION FUNCTION Z(β)")
print("="*60)
print("Energy model: E(S) = -1.5 kcal/mol per stacking pair in S")

all_results = {}
for name, db in structures:
    r = compute_rna_Z(db, name)
    if r: all_results[name] = r

print(f"\n\n{'='*60}")
print("SUMMARY: BOUNDARY DOMINANCE IN RNA")
print(f"{'='*60}")
print("""
The key prediction: terminal base pairs should have LOWER occupation
probability than interior pairs, because they contribute less
stacking energy and are more easily "removed" from the convex subset.

This is the RNA analogue of:
  - Black holes: boundary elements fluctuate, bulk doesn't
  - JT gravity: corners have lower ⟨φ⟩ than interior
  
If confirmed against SHAPE/DMS chemical probing data:
  High ⟨φ⟩ → low SHAPE reactivity (structured, protected)
  Low ⟨φ⟩  → high SHAPE reactivity (flexible, exposed)
""")
