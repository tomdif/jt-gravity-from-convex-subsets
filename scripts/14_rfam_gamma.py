"""
Compute exact γ for Rfam families with ≤ 22 structural elements.
Use the CONTAINMENT POSET (where γ > 1 for branched structures).

The structural elements are: base pairs + unpaired nucleotides.
But unpaired nucleotides inflate the element count too much.

Better: use only BASE PAIRS as elements, ordered by nesting.
Unpaired nucleotides don't participate in the partial order.
This gives a smaller poset (just the pairs), and γ measures
the branching complexity of the stem-loop architecture.
"""
import urllib.request, os, time, math
import numpy as np

CACHE = "/tmp/rfam_cache"

def download_stockholm(acc):
    cache = os.path.join(CACHE, f"{acc}.sto")
    if os.path.exists(cache) and os.path.getsize(cache) > 100:
        with open(cache) as f: return f.read()
    url = f"https://rfam.org/family/{acc}/alignment?acc={acc}&format=stockholm&download=0"
    try:
        req = urllib.request.Request(url); req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read().decode('utf-8', errors='replace')
        with open(cache, 'w') as f: f.write(data)
        return data
    except: return None

def extract_ss(sto):
    if not sto: return None
    ss = ''.join(line.split(None, 2)[2].strip() for line in sto.split('\n')
                 if line.startswith("#=GC SS_cons") and len(line.split(None, 2)) >= 3)
    simple = ""
    for c in ss:
        if c in '(<[{': simple += '('
        elif c in ')>]}': simple += ')'
        else: simple += '.'
    return simple if simple else None

def parse_pairs(db):
    stack = []; pairs = []
    for i, c in enumerate(db):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack: pairs.append((stack.pop(), i))
    return sorted(pairs)

def count_cc_containment(pairs):
    """Exact CC count on the containment poset of base pairs.
    Pair (i,j) ≤ pair (k,l) if k ≤ i and j ≤ l (i,j nested inside k,l)."""
    m = len(pairs)
    if m > 22 or m == 0: return None, None
    
    le = [[False]*m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            ia, ja = pairs[a]
            ib, jb = pairs[b]
            le[a][b] = (ib <= ia and ja <= jb)
    
    cc = 0; ints = 0
    for bits in range(1 << m):
        S = [k for k in range(m) if bits & (1 << k)]
        convex = True
        for a in S:
            if not convex: break
            for b in S:
                if not convex: break
                if le[a][b]:
                    for c in range(m):
                        if le[a][c] and le[c][b] and not (bits & (1 << c)):
                            convex = False; break
        if convex:
            cc += 1
            if len(S) <= 1:
                ints += 1
            else:
                is_int = False
                for a in S:
                    for b in S:
                        if le[a][b]:
                            iv = {c for c in range(m) if le[a][c] and le[c][b]}
                            if iv == set(S): is_int = True; break
                    if is_int: break
                if is_int: ints += 1
    return cc, ints

families = [
    ("RF00037", "IRE"), ("RF00521", "SAM_riboswitch"), ("RF00026", "U6"),
    ("RF01054", "preQ1-II"), ("RF01734", "fluoride"), ("RF01739", "glutamine"),
    ("RF00167", "Purine"), ("RF03057", "SAM-VI"), ("RF00059", "TPP"),
    ("RF00029", "Intron_gpII"), ("RF00005", "tRNA"), ("RF00050", "FMN"),
    ("RF00504", "Glycine"), ("RF01051", "c-di-GMP-I"), ("RF01750", "ZMP"),
    ("RF00002", "5_8S_rRNA"), ("RF00080", "yybP-ykoY"),
    ("RF00162", "SAM"), ("RF00379", "ydaO-yuaA"),
    ("RF00015", "U4"), ("RF00020", "U5"),
    ("RF00168", "Lysine"), ("RF01786", "c-di-AMP"),
]

print("RFAM γ ON BASE-PAIR CONTAINMENT POSET (exact)")
print("="*75)
print(f"{'Acc':>10} {'Name':>18} {'Len':>5} {'Pairs':>5} {'Depth':>5} {'|CC|':>10} {'|Int|':>6} {'γ':>8}")
print("-"*75)

results = []
for acc, name in families:
    sto = download_stockholm(acc)
    ss = extract_ss(sto)
    if not ss: continue
    
    pairs = parse_pairs(ss)
    n = len(ss)
    m = len(pairs)
    
    depth = 0; max_depth = 0
    for c in ss:
        if c == '(': depth += 1; max_depth = max(max_depth, depth)
        elif c == ')': depth -= 1
    
    if m <= 22:
        t0 = time.time()
        cc, ints = count_cc_containment(pairs)
        elapsed = time.time() - t0
        if cc and ints and ints > 0:
            gamma = cc / ints
            print(f"{acc:>10} {name[:18]:>18} {n:>5} {m:>5} {max_depth:>5} {cc:>10} {ints:>6} {gamma:>8.2f} ({elapsed:.1f}s)")
            results.append((acc, name, n, m, max_depth, cc, ints, gamma))
    else:
        print(f"{acc:>10} {name[:18]:>18} {n:>5} {m:>5} {max_depth:>5} {'skip(>22)':>10}")

if len(results) >= 3:
    print(f"\n{'='*75}")
    print(f"ANALYSIS: {len(results)} Rfam families")
    print(f"{'='*75}")
    
    gammas = np.array([r[7] for r in results])
    depths = np.array([r[4] for r in results], dtype=float)
    npairs = np.array([r[3] for r in results], dtype=float)
    
    log_g = np.log(gammas + 1)
    for feat_name, feat in [("nesting_depth", depths), ("num_pairs", npairs)]:
        if np.std(feat) > 0 and np.std(log_g) > 0:
            r = np.corrcoef(log_g, feat)[0,1]
            print(f"  log(γ+1) vs {feat_name:>15}: r = {r:.3f}")
    
    print(f"\n  γ range: [{min(gammas):.1f}, {max(gammas):.1f}]")
    print(f"\n  Sorted by γ:")
    for r in sorted(results, key=lambda x: x[7]):
        print(f"    {r[0]:>10} {r[1]:>18}: γ={r[7]:>8.2f}, pairs={r[3]:>3}, depth={r[4]:>3}")
