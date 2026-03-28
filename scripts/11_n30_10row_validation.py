"""n=30 with 10 rows (w_i sum to 30, each 1-6). More rows = finer curvature resolution."""
import numpy as np, math, time
from collections import defaultdict

def count_cc_dp(row_widths):
    m_rows = len(row_widths); max_w = max(row_widths)
    betas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    ba = np.array(betas)
    nb = len(betas)
    
    dp_ns = np.ones(nb)
    dp_c = {}; dp_g = {}
    
    for ri in range(m_rows):
        w = row_widths[ri]; nc = {}; ng = {}
        for lo in range(w):
            for hi in range(lo, w):
                ne = hi-lo+1; hl = hi-lo; ds = ne-hl
                wt = dp_ns * np.exp(-ba*ds)
                k = (lo,hi,lo); nc[k] = nc.get(k, np.zeros(nb)) + wt
        for (lp,hp,ml),wt in dp_c.items():
            ng[ml] = ng.get(ml, np.zeros(nb)) + wt
        for (lp,hp,ml),wt in dp_c.items():
            for lo in range(w):
                for hi in range(lo, w):
                    if lp <= hi:
                        if not (lo <= lp and hp >= hi): continue
                    ne=hi-lo+1; hl=hi-lo
                    vl=max(0,min(hi,hp)-max(lo,lp)+1); ds=ne-hl-vl
                    nw = wt*np.exp(-ba*ds); nm=min(ml,lo)
                    k=(lo,hi,nm); nc[k]=nc.get(k,np.zeros(nb))+nw
        for ml,wt in dp_g.items():
            ng[ml] = ng.get(ml, np.zeros(nb)) + wt
        for ml,wt in dp_g.items():
            for lo in range(w):
                for hi in range(lo, w):
                    if hi < ml:
                        ne=hi-lo+1; hl=hi-lo; ds=ne-hl
                        nw=wt*np.exp(-ba*ds); nm=min(ml,lo)
                        k=(lo,hi,nm); nc[k]=nc.get(k,np.zeros(nb))+nw
        dp_c=nc; dp_g=ng
    
    Z = dp_ns.copy()
    for wt in dp_c.values(): Z += wt
    for wt in dp_g.values(): Z += wt
    return dict(zip(betas, Z))

def R(w): return sum(w[i-1]+w[i+1]-2*w[i] for i in range(1,len(w)-1))
def LS(w):
    a=np.array(w,dtype=float)
    return sum((math.log(a[i+1])-math.log(a[i]))**2 for i in range(len(w)-1))

# Generate n=30, 10 rows, each row 1-6
print("Generating n=30, 10-row profiles...")
import random
random.seed(42)

profiles = set()
# Systematic: generate profiles with controlled curvature
for _ in range(50000):
    w = [random.randint(1,6) for _ in range(10)]
    s = sum(w)
    if s != 30: continue
    profiles.add(tuple(w))

# Also add structured profiles
for flat_w in [3]:
    profiles.add((flat_w,)*10)  # [3]*10 = 30

# Sphere-like: narrow-wide-narrow
for peak in range(4, 7):
    base = 1
    mid = [base, base+1, peak, peak, peak, peak, peak, base+1, base, base]
    s = sum(mid)
    if s == 30: profiles.add(tuple(mid))

# Saddle-like: wide-narrow-wide  
for base in range(4, 7):
    mid = [base, base-1, 2, 2, 2, 2, 2, base-1, base, base]
    s = sum(mid)
    if s == 30: profiles.add(tuple(mid))

by_R_val = defaultdict(list)
for p in profiles:
    by_R_val[R(p)].append(p)

selected = []
for Rv in sorted(by_R_val.keys()):
    ps = sorted(by_R_val[Rv])
    selected.append(ps[0])
    if len(ps) > 2: selected.append(ps[len(ps)//2])

if len(selected) > 60: selected = selected[::len(selected)//50+1][:55]

print(f"Profiles: {len(profiles)}, Selected: {len(selected)}")
print(f"R range: [{min(R(s) for s in selected)}, {max(R(s) for s in selected)}]")

t0 = time.time()
data = []
for i, w in enumerate(selected):
    Z = count_cc_dp(w)
    data.append({'w':w, 'R':R(w), 'LS':LS(w), 'Z':Z})
    if (i+1)%10==0: print(f"  {i+1}/{len(selected)} ({time.time()-t0:.1f}s)")
print(f"Done in {time.time()-t0:.1f}s")

Ra = np.array([d['R'] for d in data])
La = np.array([d['LS'] for d in data])
n = len(data)

print(f"\n{'='*60}")
print(f"JT REGRESSION: n=30, 10 rows, {n} profiles")
print(f"{'='*60}")
print(f"{'β':>6} {'R²(R)':>8} {'R²(R+LS)':>10} {'a(R)':>10} {'b(LS)':>10}")
print("-"*50)
for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
    lZ = np.array([math.log(d['Z'][beta]) for d in data])
    A1 = np.column_stack([Ra, np.ones(n)])
    c1 = np.linalg.lstsq(A1, lZ, rcond=None)[0]
    r1 = 1-np.var(lZ-A1@c1)/np.var(lZ)
    A2 = np.column_stack([Ra, La, np.ones(n)])
    c2 = np.linalg.lstsq(A2, lZ, rcond=None)[0]
    r2 = 1-np.var(lZ-A2@c2)/np.var(lZ)
    print(f"{beta:>6.1f} {r1:>8.4f} {r2:>10.4f} {c2[0]:>10.4f} {c2[1]:>10.4f}")

# Beta scaling
print(f"\nβ-scaling:")
betas_test = [0.5, 1.0, 1.5, 2.0, 3.0]
a_vals = []; b_vals = []
for beta in betas_test:
    lZ = np.array([math.log(d['Z'][beta]) for d in data])
    A = np.column_stack([Ra, La, np.ones(n)])
    c = np.linalg.lstsq(A, lZ, rcond=None)[0]
    a_vals.append(c[0]); b_vals.append(c[1])

ba = np.array(betas_test); aa = np.array(a_vals); bb = np.array(b_vals)
a_fit = np.polyfit(ba, aa, 1)
b_fit = np.polyfit(ba, bb, 1)
a_r2 = 1-np.var(aa-np.polyval(a_fit,ba))/np.var(aa)
b_r2 = 1-np.var(bb-np.polyval(b_fit,ba))/np.var(bb)
print(f"a(β) = {a_fit[0]:.4f}·β + {a_fit[1]:.4f}, R² = {a_r2:.4f}")
print(f"b(β) = {b_fit[0]:.4f}·β + {b_fit[1]:.4f}, R² = {b_r2:.4f}")

print(f"\nCOMPARISON:")
print(f"  n=15, 5 rows: R²(R+LS) = 0.846, β-linearity > 0.99")
print(f"  n=30, 5 rows: R²(R+LS) = 0.685")
print(f"  n=30, 10 rows: R²(R+LS) = see above")
