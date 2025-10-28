# Phase applied to mode 2 (squeezed light)
# Fixed robust amplitude extraction to avoid AttributeError.

import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, coherent, squeeze, destroy, qeye, tensor, fidelity

# ---------------- PARAMETERS ----------------
N_cut = 36                 # truncation for each mode 
alpha_grid = np.linspace(0.5, 6.0, 25)  # grid of coherent amplitudes to scan
r_grid = np.linspace(0.15, 0.55, 21)    # grid of squeezing strengths to scan
phi_cs = np.pi/2           # relative phase between coherent and squeezed
theta_bs = np.pi/4         # 50/50 beamsplitter mixing
N_values = list(range(2, 21))  # the total photon numbers to evaluate (2..20)
P_thresh = 1e-12           # tiny number to avoid division by zero
P_prune = 1e-9             # skip fidelity computation when P_N < this
# --------------------------------------------

# Robust amplitude helper: works if inner product returns Qobj or complex
def get_amp(bra, ket):
    val = (bra.dag() * ket)
    if hasattr(val, "full"):
        arr = val.full()
        return complex(arr.item()) if arr.size == 1 else complex(arr.ravel()[0])
    return complex(val)

# 1) Build a beam-splitter unitary acting on the two-mode space
def beam_splitter_unitary(Ncut, theta):
    a = tensor(destroy(Ncut), qeye(Ncut))
    b = tensor(qeye(Ncut), destroy(Ncut))
    H = -1j * theta * (a.dag() * b - a * b.dag())  # generator
    return H.expm()

U_bs = beam_splitter_unitary(N_cut, theta_bs)

# 2) Precompute single-mode states for speed (so we don't rebuild states inside loops)
coh_states = {alpha: coherent(N_cut, alpha) for alpha in alpha_grid}
sq_vac_states = {r: (squeeze(N_cut, r) * basis(N_cut, 0)) for r in r_grid}

# 3) Helper to build output state psi_out(alpha,r)
def build_output_state(alpha, r):
    coh = coh_states[alpha]
    sqvac = sq_vac_states[r]
    # apply relative phase to squeezed mode B: U = exp(i * phi_cs * n_b)
    n_single = destroy(N_cut).dag() * destroy(N_cut)
    Uphi_B = (1j * phi_cs * n_single).expm()
    psi_in = tensor(coh, Uphi_B * sqvac)    # two-mode input state with phase on mode 2
    psi_out = U_bs * psi_in                 # after beamsplitter
    return psi_out

# 4) Project psi_out onto subspace with total photon number Ntot, return normalized state and P_N
def project_N_component_and_prob(psi_out, Ntot):
    coeffs = []
    for m in range(N_cut):
        n = Ntot - m
        if 0 <= n < N_cut:
            ket_mn = tensor(basis(N_cut, m), basis(N_cut, n))
            amp = get_amp(ket_mn, psi_out)
            coeffs.append(((m, n), amp))
    P_N = sum(abs(c)**2 for (_, c) in coeffs)
    if P_N < P_thresh:
        return None, 0.0
    vec = sum(c * tensor(basis(N_cut, m), basis(N_cut, n)) for ((m, n), c) in coeffs)
    psi_N = vec.unit()
    return psi_N, P_N

# 5) Construct ideal NOON state for given N
def noon_state(Ntot):
    k1 = tensor(basis(N_cut, Ntot), basis(N_cut, 0))
    k2 = tensor(basis(N_cut, 0), basis(N_cut, Ntot))
    return (k1 + k2).unit()

# quick truncation sanity check 
alpha_test = alpha_grid[len(alpha_grid)//2]
r_test = r_grid[len(r_grid)//2]
psi_out_test = build_output_state(alpha_test, r_test)
totalP = 0.0
for m in range(N_cut):
    for n in range(N_cut):
        ket_mn = tensor(basis(N_cut, m), basis(N_cut, n))
        amp = get_amp(ket_mn, psi_out_test)
        totalP += abs(amp)**2
print(f"Sanity check: total probability in truncation = {totalP:.6f} (should be near 1)")

# 6) main optimization loop
start_time = time.time()
best_F = []
best_params = []
best_Pn = []
total_iters = len(N_values) * len(alpha_grid) * len(r_grid)
iter_count = 0
print("Starting parameter scan...")

for Ntot in N_values:
    bestF = -1.0
    best_pair = (None, None)
    bestP = 0.0
    for alpha in alpha_grid:
        for r in r_grid:
            iter_count += 1
            psi_out = build_output_state(alpha, r)
            psi_N, P_N = project_N_component_and_prob(psi_out, Ntot)
            if psi_N is None:
                continue
            if P_N < P_prune:
                continue
            noonN = noon_state(Ntot)
            F = fidelity(noonN, psi_N)
            if F > bestF:
                bestF = F
                best_pair = (alpha, r)
                bestP = P_N
    if bestF < 0:
        bestF = float('nan')
    best_F.append(bestF)
    best_params.append(best_pair)
    best_Pn.append(bestP)
    print(f"[{iter_count}/{total_iters}] N={Ntot} best F={bestF:.6f} alpha={best_pair[0]} r={best_pair[1]} P_N={bestP:.3e}")

print("Optimization done in {:.1f} s".format(time.time() - start_time))

# 7) plot the best fidelities vs N
plt.figure(figsize=(8,4))
plt.plot(N_values, best_F, 'o-')
plt.xlabel('Total photon number N')
plt.ylabel('Optimized fidelity F_N')
plt.title('Optimized NOON fidelities (coarse grid)')
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

# 8) print safe summary
print("\nSummary per N:")
for Ntot, best_pair, Pn, Fval in zip(N_values, best_params, best_Pn, best_F):
    if best_pair[0] is None:
        print(f"N={Ntot}: no feasible parameters found (pruned or P_N ~ 0)")
    else:
        a_best, r_best = best_pair
        print(f"N={Ntot}: alpha={a_best:.3f}, r={r_best:.3f}, P_N={Pn:.3e}, F={Fval:.6f}")
