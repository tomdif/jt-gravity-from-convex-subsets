"""
Noetherian ratio γ = |CC|/|Int| for chemical reaction network posets.

CC = convex subsets (order-convex / intermediate-closed)
Int = intervals [a,b] = {c : a ≤ c ≤ b} (including singletons [a,a] and empty set)
"""

from itertools import combinations, product
from collections import deque

def transitive_closure(nodes, edges):
    """Compute transitive closure of a DAG. Returns dict: node -> set of reachable nodes."""
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)

    # Floyd-Warshall style, or BFS from each node
    reach = {n: set() for n in nodes}
    for start in nodes:
        visited = set()
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    queue.append(nxt)
        reach[start] = visited  # includes start itself
    return reach

def leq(reach, a, b):
    """a ≤ b in the poset iff b is reachable from a."""
    return b in reach[a]

def is_convex(subset_set, reach, nodes):
    """Check if subset is order-convex: for all a,b in S with a≤b, all c with a≤c≤b are in S."""
    for a in subset_set:
        for b in subset_set:
            if a == b:
                continue
            if leq(reach, a, b):
                # Check all c in nodes \ S
                for c in nodes:
                    if c not in subset_set:
                        if leq(reach, a, c) and leq(reach, c, b):
                            return False
    return True

def count_intervals(reach, nodes):
    """
    Count intervals [a,b] = {c : a ≤ c ≤ b}.
    Include: empty set, all singletons [a,a], and all [a,b] for a ≤ b.
    Two different pairs (a,b) and (a',b') giving the same set count as ONE interval.
    """
    intervals = set()
    # Empty set
    intervals.add(frozenset())

    for a in nodes:
        for b in nodes:
            if leq(reach, a, b):
                # interval [a,b]
                iv = frozenset(c for c in nodes if leq(reach, a, c) and leq(reach, c, b))
                intervals.add(iv)
    return intervals

def count_convex_subsets(reach, nodes):
    """Brute force: enumerate all 2^|V| subsets, check convexity."""
    n = len(nodes)
    node_list = list(nodes)
    convex = []
    for mask in range(1 << n):
        subset = set()
        for i in range(n):
            if mask & (1 << i):
                subset.add(node_list[i])
        if is_convex(subset, reach, nodes):
            convex.append(frozenset(subset))
    return convex

def compute_gamma(name, nodes, edges):
    """Compute and report γ for a reaction network."""
    nodes = list(nodes)
    reach = transitive_closure(nodes, edges)

    print(f"\n{'='*60}")
    print(f"Network: {name}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")

    # Count intervals (as distinct sets)
    intervals = count_intervals(reach, nodes)
    num_int = len(intervals)

    # Count convex subsets
    convex = count_convex_subsets(reach, nodes)
    num_cc = len(convex)

    gamma = num_cc / num_int if num_int > 0 else float('inf')

    print(f"|CC| (convex subsets): {num_cc}")
    print(f"|Int| (distinct intervals): {num_int}")
    print(f"γ = |CC|/|Int| = {gamma:.6f}")
    print(f"2^|V| = {1 << len(nodes)}")
    print(f"|CC|/2^|V| = {num_cc / (1 << len(nodes)):.6f}")

    # Show which convex subsets are NOT intervals
    non_interval_convex = [s for s in convex if s not in intervals]
    if len(non_interval_convex) <= 20:
        if non_interval_convex:
            print(f"Convex but not interval ({len(non_interval_convex)}):")
            for s in sorted(non_interval_convex, key=lambda x: (len(x), sorted(x))):
                print(f"  {set(s)}")
    else:
        print(f"Convex but not interval: {len(non_interval_convex)} sets")

    return gamma, num_cc, num_int

# ============================================================
# 1. GLYCOLYSIS (linear chain, 10 nodes)
# ============================================================
glycolysis_nodes = ["Glc", "G6P", "F6P", "FBP", "G3P", "BPG", "3PG", "2PG", "PEP", "Pyr"]
glycolysis_edges = [(glycolysis_nodes[i], glycolysis_nodes[i+1]) for i in range(9)]

# ============================================================
# 2. TCA CYCLE (broken as DAG: OAA_start → ... → OAA_end)
# We treat it as a linear chain with OAA appearing twice
# ============================================================
tca_nodes = ["OAA_s", "Cit", "IsoCit", "aKG", "SucCoA", "Succ", "Fum", "Mal", "OAA_e"]
tca_edges = [(tca_nodes[i], tca_nodes[i+1]) for i in range(8)]

# ============================================================
# 3. BRANCHING: Pyruvate → {Acetyl-CoA, Lactate, Alanine, Ethanol}
# ============================================================
branch_nodes = ["Pyr", "AcCoA", "Lac", "Ala", "EtOH"]
branch_edges = [("Pyr", "AcCoA"), ("Pyr", "Lac"), ("Pyr", "Ala"), ("Pyr", "EtOH")]

# ============================================================
# 4. Y-JUNCTION: {A, B} → C → {D, E}
# ============================================================
yjunc_nodes = ["A", "B", "C", "D", "E"]
yjunc_edges = [("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")]

# ============================================================
# 5. DIAMOND: A → {B, C} → D
# ============================================================
diamond_nodes = ["A", "B", "C", "D"]
diamond_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

# ============================================================
# Run all small networks
# ============================================================
results = {}

for name, nodes, edges in [
    ("Glycolysis (linear chain, 10)", glycolysis_nodes, glycolysis_edges),
    ("TCA cycle (broken DAG, 9)", tca_nodes, tca_edges),
    ("Branching (1→4)", branch_nodes, branch_edges),
    ("Y-junction ({A,B}→C→{D,E})", yjunc_nodes, yjunc_edges),
    ("Diamond (A→{B,C}→D)", diamond_nodes, diamond_edges),
]:
    g, cc, iv = compute_gamma(name, nodes, edges)
    results[name] = (g, cc, iv, len(nodes))

# ============================================================
# 6. Theoretical check: linear chain of length n
# For a linear chain of n nodes, intervals = 1 + n(n+1)/2
# (empty set + n singletons + C(n,2) proper intervals)
# Convex subsets of a linear order = intervals (unions of
# consecutive elements + empty set)
# So γ should be exactly 1.
# ============================================================
print(f"\n{'='*60}")
print("VERIFICATION: Linear chains")
for n in [3, 5, 7, 10]:
    nodes = list(range(n))
    edges = [(i, i+1) for i in range(n-1)]
    reach = transitive_closure(nodes, edges)
    intervals = count_intervals(reach, nodes)
    convex = count_convex_subsets(reach, nodes)
    print(f"  Chain({n}): |CC|={len(convex)}, |Int|={len(intervals)}, γ={len(convex)/len(intervals):.4f}")
    expected_int = 1 + n*(n+1)//2
    print(f"    Expected |Int| = 1 + n(n+1)/2 = {expected_int}, got {len(intervals)}")

# ============================================================
# 7. ANTICHAIN check: n incomparable elements
# Every subset is convex. Intervals = empty + n singletons = n+1.
# γ = 2^n / (n+1)
# ============================================================
print(f"\n{'='*60}")
print("VERIFICATION: Antichains")
for n in [2, 3, 4, 5]:
    nodes = list(range(n))
    edges = []
    reach = transitive_closure(nodes, edges)
    intervals = count_intervals(reach, nodes)
    convex = count_convex_subsets(reach, nodes)
    print(f"  Antichain({n}): |CC|={len(convex)}, |Int|={len(intervals)}, γ={len(convex)/len(intervals):.4f}")

# ============================================================
# 8. E. COLI CENTRAL CARBON METABOLISM (simplified DAG, ~22 nodes)
# Glycolysis + Pentose Phosphate + TCA
# ============================================================
print(f"\n{'='*60}")
print("Building E. coli central carbon metabolism (simplified)...")

ecoli_nodes = [
    # Glycolysis
    "Glc", "G6P", "F6P", "FBP", "G3P", "BPG", "3PG", "2PG", "PEP", "Pyr",
    # Pentose phosphate pathway
    "6PGL", "6PG", "Ru5P", "R5P", "X5P", "S7P", "E4P",
    # TCA (as DAG)
    "AcCoA", "Cit", "aKG", "SucCoA", "OAA"
]

ecoli_edges = [
    # Glycolysis
    ("Glc", "G6P"), ("G6P", "F6P"), ("F6P", "FBP"), ("FBP", "G3P"),
    ("G3P", "BPG"), ("BPG", "3PG"), ("3PG", "2PG"), ("2PG", "PEP"),
    ("PEP", "Pyr"),
    # Pyruvate → Acetyl-CoA
    ("Pyr", "AcCoA"),
    # TCA (linear portion)
    ("AcCoA", "Cit"), ("Cit", "aKG"), ("aKG", "SucCoA"),
    # OAA feeds into citrate (with AcCoA), but as DAG: OAA → Cit already covered
    # PEP → OAA (anaplerotic)
    ("PEP", "OAA"),
    ("OAA", "Cit"),
    # Pentose phosphate pathway
    ("G6P", "6PGL"), ("6PGL", "6PG"), ("6PG", "Ru5P"),
    ("Ru5P", "R5P"), ("Ru5P", "X5P"),
    # Transketolase/transaldolase connections
    ("R5P", "S7P"), ("X5P", "G3P"),  # TKT1: R5P + X5P → S7P + G3P
    ("S7P", "E4P"),                    # TAL: S7P + G3P → E4P + F6P
    ("E4P", "F6P"),                    # via transaldolase
    ("X5P", "BPG"),                    # TKT2: X5P + E4P → F6P + G3P (simplified)
]

# Check for cycles before computing
print(f"E. coli nodes: {len(ecoli_nodes)}")
print(f"E. coli edges: {len(ecoli_edges)}")

# Verify it's a DAG (topological sort exists)
from collections import defaultdict

def is_dag(nodes, edges):
    adj = defaultdict(set)
    indeg = defaultdict(int)
    for n in nodes:
        indeg[n] = 0
    for a, b in edges:
        adj[a].add(b)
        indeg[b] += 1
    queue = deque([n for n in nodes if indeg[n] == 0])
    count = 0
    while queue:
        n = queue.popleft()
        count += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return count == len(nodes)

if is_dag(ecoli_nodes, ecoli_edges):
    print("Verified: DAG (no cycles)")
    g, cc, iv = compute_gamma("E. coli central carbon (22 nodes)", ecoli_nodes, ecoli_edges)
    results["E. coli central carbon (22 nodes)"] = (g, cc, iv, len(ecoli_nodes))
else:
    print("WARNING: Graph has cycles! Checking...")
    # Find cycle
    reach = transitive_closure(ecoli_nodes, ecoli_edges)
    for n in ecoli_nodes:
        for m in reach[n]:
            if m != n and n in reach[m]:
                print(f"  Cycle: {n} → ... → {m} → ... → {n}")
                break

# ============================================================
# 9. SUMMARY TABLE
# ============================================================
print(f"\n{'='*60}")
print(f"{'='*60}")
print("SUMMARY: Noetherian Ratios for Chemical Reaction Networks")
print(f"{'='*60}")
print(f"{'Network':<45} {'|V|':>4} {'|CC|':>8} {'|Int|':>8} {'γ':>10}")
print(f"{'-'*45} {'-'*4} {'-'*8} {'-'*8} {'-'*10}")
for name, (g, cc, iv, n) in results.items():
    print(f"{name:<45} {n:>4} {cc:>8} {iv:>8} {g:>10.4f}")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print(f"{'='*60}")
print("""
Key findings:
- Linear chains (glycolysis, TCA broken): γ = 1.0000
  Every convex subset IS an interval. This is a theorem for
  totally ordered sets.

- Branching (1→4): γ > 1. The 4 leaves are pairwise
  incomparable, creating convex subsets that aren't intervals
  (e.g., {Lac, Ala} is convex but not [a,b] for any a,b).

- Y-junction and Diamond: γ > 1 due to parallel paths creating
  incomparable elements.

- E. coli metabolism: γ reflects the combined branching and
  merging in central carbon metabolism.

The Noetherian ratio γ measures "how far from a total order"
a poset is. γ = 1 iff the poset is a total order (linear chain).
Branching and merging increase γ by creating incomparable
elements whose subsets are vacuously convex.

For an antichain of n elements: γ = 2^n/(n+1), which grows
exponentially — maximum possible divergence from linearity.
""")
