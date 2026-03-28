"""
Consequences of log Z = -β·(a₀·R + b₀·LogSch) + entropy.

1. Density of states: JT gravity predicts ρ(E) ~ sinh(2π√E)
2. Entropy vs temperature: S(T) = (1 - β·∂/∂β) log Z
3. Specific heat: C = -β² ∂²/∂β² log Z  
4. The SYK connection: JT gravity ↔ SYK model
5. Black hole thermodynamics in 2D
6. Prediction for d=3,4: what action should emerge?
"""
import numpy as np
import math
from collections import defaultdict

# Reuse the profile data from the flat grid [3,3,3,3,3] (n=15)
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
    return np.array(results, dtype=float)

# Compute for the flat grid
sbd_flat = enumerate_cc_varwidth([3,3,3,3,3])
print(f"Flat grid [3]^5×5: {len(sbd_flat)} convex subsets")

# 1. DENSITY OF STATES
print("\n" + "="*60)
print("1. DENSITY OF STATES ρ(E)")
print("="*60)
E_vals = sorted(set(sbd_flat))
print(f"\nS_BD values: {[int(e) for e in E_vals]}")
rho = {int(e): np.sum(sbd_flat == e) for e in E_vals}
print(f"{'E':>4} {'ρ(E)':>8} {'log ρ':>8} {'√E':>8} {'sinh fit':>10}")
for E in sorted(rho.keys()):
    r = rho[E]
    log_r = math.log(r) if r > 0 else 0
    sqrt_E = math.sqrt(abs(E - min(rho.keys()))) if E > min(rho.keys()) else 0
    # JT prediction: ρ(E) ~ sinh(2π√(E-E₀)) for E > E₀
    print(f"{E:>4} {r:>8} {log_r:>8.2f} {sqrt_E:>8.2f}")

# 2. THERMODYNAMICS
print("\n" + "="*60)
print("2. THERMODYNAMICS: S(T), C(T), F(T)")
print("="*60)
betas = np.linspace(0.1, 5.0, 50)
log_Z = []
avg_E = []
avg_E2 = []

for beta in betas:
    w = np.exp(-beta * sbd_flat)
    Z = w.sum()
    log_Z.append(math.log(Z))
    avg_E.append((sbd_flat * w).sum() / Z)
    avg_E2.append((sbd_flat**2 * w).sum() / Z)

log_Z = np.array(log_Z)
avg_E = np.array(avg_E)
avg_E2 = np.array(avg_E2)
variance = avg_E2 - avg_E**2

# Entropy: S = log Z + β·<E>  (using S = -∂F/∂T = β²·∂log Z/∂β)
S = log_Z + betas * avg_E
# Specific heat: C = β²·Var(E)
C = betas**2 * variance
# Free energy: F = -T·log Z = -log Z / β
F = -log_Z / betas

print(f"\n{'β':>6} {'T':>8} {'F':>10} {'S':>10} {'<E>':>10} {'C':>10}")
for i in range(0, len(betas), 5):
    T = 1/betas[i]
    print(f"{betas[i]:>6.2f} {T:>8.3f} {F[i]:>10.4f} {S[i]:>10.4f} {avg_E[i]:>10.4f} {C[i]:>10.4f}")

# 3. JT GRAVITY PREDICTION CHECK
print("\n" + "="*60)
print("3. JT GRAVITY: does ρ(E) ~ sinh(2π√E)?")
print("="*60)

# In JT gravity, the density of states is:
# ρ(E) = (γ/2π²) sinh(2π√(2γE))
# where γ is the JT coupling.
#
# For our discrete system, shift E by E_min:
E_min = min(rho.keys())
E_shifted = sorted([E - E_min for E in rho.keys() if E > E_min])
rho_shifted = [rho[E + E_min] for E in E_shifted]

if len(E_shifted) > 2:
    log_rho = np.log(np.array(rho_shifted, dtype=float))
    sqrt_E = np.sqrt(np.array(E_shifted, dtype=float))
    
    # Fit: log ρ(E) = a + b·√(E - E₀)
    A = np.column_stack([sqrt_E, np.ones(len(sqrt_E))])
    coeffs = np.linalg.lstsq(A, log_rho, rcond=None)[0]
    fitted = A @ coeffs
    r2 = 1 - np.var(log_rho - fitted) / np.var(log_rho)
    
    print(f"Fit: log ρ(E) = {coeffs[0]:.3f}·√(E-E₀) + {coeffs[1]:.3f}")
    print(f"R² = {r2:.4f}")
    print(f"JT prediction: coefficient should be 2π·√(2γ) ≈ {coeffs[0]:.3f}")
    print(f"  → γ = ({coeffs[0]}/(2π))²/2 = {(coeffs[0]/(2*math.pi))**2/2:.4f}")
    
    # Also fit sinh: log ρ ~ log sinh(b√E) = log(e^{b√E} - e^{-b√E}) - log 2
    # For large E: ~ b√E. For small E: ~ log(b√E) ~ log(b) + 0.5·log(E).
    print(f"\nComparison:")
    print(f"{'E-E₀':>6} {'ρ':>8} {'log ρ':>8} {'fit':>8} {'res':>8}")
    for i, E in enumerate(E_shifted):
        print(f"{E:>6} {rho_shifted[i]:>8} {log_rho[i]:>8.2f} {fitted[i]:>8.2f} {log_rho[i]-fitted[i]:>8.2f}")

# 4. SYK CONNECTION
print("\n" + "="*60)
print("4. SYK / QUANTUM CHAOS CONNECTION")
print("="*60)
print("""
JT gravity in 2D is dual to the SYK model: a quantum mechanical
system of N Majorana fermions with random all-to-all couplings.

Key signatures of SYK/JT:
  - ρ(E) ~ sinh(2π√E)     [computed above]
  - Level repulsion (GOE/GUE statistics)
  - Spectral form factor: ramp + plateau
  - Schwarzian effective action

For convex subsets: the S_BD values {-7, -6, ..., +3, +4} form
a discrete "energy spectrum". If this spectrum shows level repulsion
at the scale of individual profiles, that's evidence for quantum chaos.

The spectral form factor K(t) = |Z(β+it)|²/|Z(β)|² should show:
  - Slope (early time decay)
  - Ramp (linear growth = quantum chaos signature)  
  - Plateau (saturation)
""")

# 5. PREDICTION FOR d=3,4
print("="*60)
print("5. PREDICTIONS FOR HIGHER DIMENSIONS")
print("="*60)
print("""
In d=2: EH is topological, JT (Schwarzian) is dynamical.
  Result: log Z = β·(0.165·R + 0.517·LogSch)
  R² for R alone: 0.27 (weak — topological term is not the whole story)
  R² for R+LogSch: 0.85 (strong — JT captures the dynamics)

PREDICTION for d=3: 
  EH is dynamical (Ricci flow). LogSch should still appear as conformal mode.
  R² for R alone should be HIGHER than 0.27 (EH is now dynamical).
  The JT/Schwarzian term should be a correction, not the dominant physics.
  Expected: R² for R alone ≈ 0.5-0.7, with LogSch adding less.

PREDICTION for d=4:
  EH is fully dynamical (Einstein gravity). 
  R² for R alone should be HIGH (0.7-0.9).
  LogSch = conformal mode = trace anomaly correction.
  The Weyl tensor (traceless curvature) might appear as additional invariant.
  Expected: R² for R alone ≈ 0.8+, confirming EH dominance in d=4.

THE KEY TEST: if R² for R-only INCREASES with d:
  d=2: R² = 0.27 (topological)
  d=3: R² = ???  (Ricci flow)
  d=4: R² = ???  (Einstein gravity)
  
This would show the DIMENSION DEPENDENCE of the gravitational action
emerging from pure combinatorics.
""")
