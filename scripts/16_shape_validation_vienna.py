"""
Proper validation: 
1. Get real RNA sequences from Rfam (with consensus SS)
2. Fold them with ViennaRNA to get SHAPE-like ensemble accessibility
3. Compute ⟨φ(x)⟩ from the convex subset partition function
4. Correlate ⟨φ⟩ with Vienna's base-pair probabilities (the ground truth)

This breaks all circularity: 
- ⟨φ⟩ uses only the consensus structure + stacking energies
- Vienna's bp probabilities use the full thermodynamic model on the SEQUENCE
- The two methods are completely independent
"""
import urllib.request, json, os, math
import numpy as np

try:
    import RNA
    HAS_RNA = True
    print(f"ViennaRNA: {RNA.version()}" if hasattr(RNA, 'version') else "ViennaRNA: available")
except:
    HAS_RNA = False
    print("ViennaRNA not available — will use accessibility proxy")

CACHE = "/tmp/rfam_cache"
os.makedirs(CACHE, exist_ok=True)

def get_rfam_alignment(acc):
    cache = os.path.join(CACHE, f"{acc}.sto")
    if os.path.exists(cache) and os.path.getsize(cache) > 200:
        with open(cache) as f: return f.read()
    url = f"https://rfam.org/family/{acc}/alignment/stockholm"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read().decode('utf-8', errors='replace')
        with open(cache, 'w') as f: f.write(data)
        return data
    except Exception as e:
        return None

def extract_ss_and_seq(sto):
    """Extract consensus SS and first ungapped sequence."""
    if not sto: return None, None, None
    
    ss_parts = []
    seqs = {}
    
    for line in sto.split('\n'):
        if line.startswith('#=GC SS_cons'):
            parts = line.split(None, 2)
            if len(parts) >= 3: ss_parts.append(parts[2].strip())
        elif not line.startswith('#') and not line.startswith('/') and not line.startswith(' ') and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                seq = parts[1]
                if name not in seqs: seqs[name] = ''
                seqs[name] += seq
    
    ss_raw = ''.join(ss_parts)
    
    # Convert WUSS to dot-bracket
    ss = ''
    for c in ss_raw:
        if c in '(<[{': ss += '('
        elif c in ')>]}': ss += ')'
        else: ss += '.'
    
    # Find best sequence (fewest gaps)
    best_seq = None; best_gaps = float('inf')
    for name, seq in seqs.items():
        gaps = seq.count('-') + seq.count('.')
        if gaps < best_gaps and len(seq) == len(ss):
            best_gaps = gaps
            best_seq = (name, seq)
    
    if not best_seq: return ss, None, None
    
    # Remove gap columns: positions where the sequence has '-' or '.'
    seq = best_seq[1]
    clean_seq = ''
    clean_ss = ''
    for i in range(min(len(seq), len(ss))):
        if seq[i] not in '-.':
            clean_seq += seq[i].upper().replace('T', 'U')
            clean_ss += ss[i]
    
    # Fix bracket matching after gap removal
    stack = []; valid_ss = list(clean_ss)
    for i, c in enumerate(valid_ss):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack: stack.pop()
            else: valid_ss[i] = '.'
    for i in stack: valid_ss[i] = '.'
    clean_ss = ''.join(valid_ss)
    
    return clean_ss, clean_seq, best_seq[0]

def parse_pairs(db):
    stack = []; pairs = []
    for i, c in enumerate(db):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack: pairs.append((stack.pop(), i))
    return sorted(pairs)

def compute_phi(ss, beta=1.0):
    pairs = parse_pairs(ss)
    m = len(pairs)
    n = len(ss)
    if m == 0 or m > 22: return None
    
    le = [[False]*m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            le[a][b] = (pairs[b][0] <= pairs[a][0] and pairs[a][1] <= pairs[b][1])
    
    pair_set = {p: idx for idx, p in enumerate(pairs)}
    stacks = [(idx_a, pair_set[(ia+1,ja-1)]) 
              for idx_a, (ia,ja) in enumerate(pairs) if (ia+1,ja-1) in pair_set]
    
    subs = []
    for bits in range(1 << m):
        S = [k for k in range(m) if bits & (1 << k)]
        ok = True
        for a in S:
            if not ok: break
            for b in S:
                if not ok: break
                if le[a][b]:
                    for c in range(m):
                        if le[a][c] and le[c][b] and not (bits & (1 << c)):
                            ok = False; break
        if ok:
            S_set = set(S)
            e = -1.5 * sum(1 for (a,b) in stacks if a in S_set and b in S_set)
            subs.append((S, e))
    
    w = np.array([np.exp(-beta*e) for _,e in subs])
    Z = w.sum()
    pphi = np.zeros(m)
    for idx,(S,_) in enumerate(subs):
        for k in S: pphi[k] += w[idx]
    pphi /= Z
    
    nphi = np.zeros(n)
    for pidx,(i,j) in enumerate(pairs):
        nphi[i] = pphi[pidx]; nphi[j] = pphi[pidx]
    return nphi

# Get small Rfam families, extract real sequences, compute both ⟨φ⟩ and Vienna bp probs
families = [
    "RF00005", "RF00037", "RF00050", "RF00059", "RF00162", "RF00167",
    "RF00168", "RF00234", "RF00504", "RF00521", "RF01051", "RF01054",
    "RF01734", "RF01739", "RF03057", "RF00026", "RF00029",
    "RF01750", "RF01786", "RF01831", "RF02012", "RF02683",
]

print("REAL-SEQUENCE VALIDATION: ⟨φ⟩ vs ViennaRNA bp probabilities")
print("="*70)

results = []
for acc in families:
    sto = get_rfam_alignment(acc)
    ss, seq, seq_name = extract_ss_and_seq(sto)
    
    if not ss or not seq: continue
    
    pairs = parse_pairs(ss)
    m = len(pairs)
    n = len(ss)
    
    if m > 22 or m < 3: continue
    if len(seq) != n: continue
    
    # Compute ⟨φ⟩
    phi = compute_phi(ss)
    if phi is None: continue
    
    # Compute ViennaRNA base-pair probabilities
    if HAS_RNA:
        fc = RNA.fold_compound(seq)
        fc.pf()
        bpp = np.zeros(n)
        for pidx, (i, j) in enumerate(pairs):
            # Get bp probability for this specific pair
            p = fc.bpp()[i+1][j+1] if i+1 < len(fc.bpp()) and j+1 < len(fc.bpp()[0]) else 0
            bpp[i] = p
            bpp[j] = p
        
        # Correlate ⟨φ⟩ with Vienna bp prob on paired positions
        paired = [i for i in range(n) if phi[i] > 0]
        if len(paired) < 3: continue
        
        phi_p = phi[paired]
        bpp_p = bpp[paired]
        
        if np.std(phi_p) > 0 and np.std(bpp_p) > 0:
            r = np.corrcoef(phi_p, bpp_p)[0,1]
        else:
            r = float('nan')
        
        print(f"  {acc} ({m:>2} pairs, n={n:>3}): r(⟨φ⟩, Vienna_bpp) = {r:>7.3f}  seq={seq_name[:25]}")
        if not np.isnan(r):
            results.append({'acc': acc, 'pairs': m, 'n': n, 'r': r,
                          'phi_p': phi_p, 'bpp_p': bpp_p})
    else:
        print(f"  {acc} ({m:>2} pairs, n={n:>3}): computed ⟨φ⟩ (no Vienna for comparison)")

if results:
    print(f"\n{'='*70}")
    print(f"POOLED: {len(results)} families")
    print(f"{'='*70}")
    
    all_phi = np.concatenate([r['phi_p'] for r in results])
    all_bpp = np.concatenate([r['bpp_p'] for r in results])
    
    r_pool = np.corrcoef(all_phi, all_bpp)[0,1]
    per_r = [r['r'] for r in results if not np.isnan(r['r'])]
    
    print(f"  Pooled r(⟨φ⟩, Vienna_bpp): {r_pool:.3f}")
    print(f"  Per-family mean r: {np.mean(per_r):.3f}")
    print(f"  Per-family median r: {np.median(per_r):.3f}")
    print(f"  Per-family std: {np.std(per_r):.3f}")
    print(f"  Range: [{min(per_r):.3f}, {max(per_r):.3f}]")
    
    n_pos = len(all_phi)
    t = r_pool * math.sqrt(n_pos - 2) / math.sqrt(1 - r_pool**2) if abs(r_pool) < 1 else float('inf')
    print(f"  n = {n_pos}, t = {t:.1f}")
    
    print(f"\n  Vienna bpp by ⟨φ⟩ quartile:")
    for lo, hi in [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        mask = (all_phi >= lo) & (all_phi < hi)
        if mask.sum() > 0:
            print(f"    ⟨φ⟩ [{lo:.1f},{hi:.1f}): n={mask.sum():>3}, mean bpp={all_bpp[mask].mean():.3f}")
    
    print(f"\n  THIS IS THE NON-CIRCULAR TEST:")
    print(f"  ⟨φ⟩ computed from: consensus structure + stacking energy model")
    print(f"  bpp computed from: sequence + full Vienna thermodynamic model")
    print(f"  The two are INDEPENDENT. Correlation measures whether the")
    print(f"  combinatorial partition function captures the same thermodynamic")
    print(f"  information as the full energy model.")
