# Simplified annotated script to compute and plot joint photon-number distribution P(m,n).
# Fixed robust amplitude extraction to avoid AttributeError.

from qutip import basis, coherent, squeeze, destroy, qeye, tensor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # added for 3D plotting
from matplotlib import cm
from matplotlib.colors import Normalize

# ---------------- PARAMETERS ----------------
N = 20              # Fock basis cutoff per single mode (0..N-1)
alpha = 1.4         # coherent amplitude (mean photons ~ alpha^2)
r = 0.7           # squeezing parameter (small positive number)
phi_rel = np.pi/2   # relative phase between coherent and squeezed input
theta = np.pi/4     # 50/50 beamsplitter mixing parameter
# --------------------------------------------

# Robust amplitude helper: works if inner product returns Qobj or complex
def get_amp(bra, ket):
    val = (bra.dag() * ket)
    # If val is a Qobj it has .full(), otherwise it might already be a complex
    if hasattr(val, "full"):
        arr = val.full()
        # arr could be 1x1 matrix
        return complex(arr.item()) if arr.size == 1 else complex(arr.ravel()[0])
    # otherwise try to coerce to complex
    return complex(val)

# 1) build single-mode objects
vac = basis(N, 0)                 # vacuum state |0>
coh = coherent(N, alpha)          # coherent state |alpha> in mode A
squeezed_vac = squeeze(N, r) * vac  # squeezed vacuum in mode B

# 2) apply a relative phase to mode B (multiply each |n> component by e^{i n phi})
n_single = destroy(N).dag() * destroy(N)  # number operator n = a^† a
Uphi_single = (1j * phi_rel * n_single).expm()  # exp(i * phi * n)
Uphi_two_mode = tensor(qeye(N), Uphi_single)   # apply to mode B only

# 3) build two-mode input states
psi_in_quantum = Uphi_two_mode * tensor(coh, squeezed_vac)  # coherent + squeezed
psi_in_classical = tensor(coh, vac)                        # coherent + vacuum

# 4) build a beamsplitter unitary (simple small-N version)
def two_mode_bs(Ndim, theta_bs):
    a = tensor(destroy(Ndim), qeye(Ndim))
    b = tensor(qeye(Ndim), destroy(Ndim))
    H = -1j * theta_bs * (a.dag() * b - a * b.dag())
    return H.expm()

U_bs = two_mode_bs(N, theta)

# 5) apply beamsplitter
psi_out_q = U_bs * psi_in_quantum
psi_out_c = U_bs * psi_in_classical

# 6) compute joint photon number distribution P(m,n)
def joint_distribution(psi_out, cutoff):
    P = np.zeros((cutoff, cutoff))
    for m in range(cutoff):
        for n in range(cutoff):
            ket_mn = tensor(basis(cutoff, m), basis(cutoff, n))
            amp = get_amp(ket_mn, psi_out)  # robust amplitude
            P[m, n] = abs(amp)**2
    total = P.sum()
    if total > 0:
        P = P / total
    return P, total

Pq, tot_q = joint_distribution(psi_out_q, N)
Pc, tot_c = joint_distribution(psi_out_c, N)

print(f"Total prob in truncated basis (quantum input) = {tot_q:.6f}")
print(f"Total prob in truncated basis (classical input) = {tot_c:.6f}")

# 7) Plot histograms (3D bars) 

cutoff_display = 15                # show low photon numbers up to 0..14
m_vals = np.arange(cutoff_display)
n_vals = np.arange(cutoff_display)
m_grid, n_grid = np.meshgrid(m_vals, n_vals, indexing='ij')

#  P is P[m,n] with m rows and n columns. 
# We transpose here so the visual matches better.
Pq_plot = Pq[:cutoff_display, :cutoff_display].T.copy()
Pc_plot = Pc[:cutoff_display, :cutoff_display].T.copy()

xpos = m_grid.ravel()   # photon number axis 1 (will appear along x)
ypos = n_grid.ravel()   # photon number axis 2 (will appear along y)
zpos = np.zeros_like(xpos)
dx = dy = 0.9           # slightly larger bars, nearly touching

# Use linear heights but use color mapping and a slight vertical exaggeration
z_q = Pq_plot.ravel()
z_c = Pc_plot.ravel()

# vertical exaggeration factor to make small probabilities visible 
vz_scale = 1.0 / z_q.max() if z_q.max() > 0 else 1.0
vz_scale_c = 1.0 / z_c.max() if z_c.max() > 0 else 1.0
# choose modest exaggeration so shape is visible but not distorted
z_q_vis = z_q * vz_scale * 0.9
z_c_vis = z_c * vz_scale_c * 0.9

# color mapping based on raw probability (not the visual scaled heights)
cmap = cm.get_cmap('viridis')
norm_q = Normalize(vmin=z_q.min(), vmax=z_q.max())
norm_c = Normalize(vmin=z_c.min(), vmax=z_c.max())
facecolors_q = cmap(norm_q(z_q))
facecolors_c = cmap(norm_c(z_c))

fig = plt.figure(figsize=(14, 6))

# --- Quantum case (Fig. 1A style)
ax = fig.add_subplot(1, 2, 1, projection='3d')
# draw bars with color mapping and black edges for clarity
ax.bar3d(xpos, ypos, zpos, dx, dy, z_q_vis, shade=True,
         color=facecolors_q, edgecolor='k', linewidth=0.2)
ax.set_title('Quantum: coherent + squeezed', fontsize=11, fontweight='bold')
ax.set_xlabel('m (mode 1)')   # match notation: m,n photon numbers
ax.set_ylabel('n (mode 2)')
ax.set_zlabel('Relative P(m,n)')
ax.view_init(elev=25, azim=-60)   # camera similar to paper
ax.set_box_aspect((cutoff_display, cutoff_display, 0.8 * cutoff_display * 0.4))

# set integer ticks for photon axes
ax.set_xticks(np.arange(0, cutoff_display, 2))
ax.set_yticks(np.arange(0, cutoff_display, 2))
# z-limits chosen to emphasize structure
ax.set_zlim(0, z_q_vis.max() * 1.05)

# --- Classical case (Fig. 1B style)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.bar3d(xpos, ypos, zpos, dx, dy, z_c_vis, shade=True,
          color=facecolors_c, edgecolor='k', linewidth=0.2)
ax2.set_title('Classical: coherent + vacuum', fontsize=11, fontweight='bold')
ax2.set_xlabel('m (mode 1)')
ax2.set_ylabel('n (mode 2)')
ax2.set_zlabel('Relative P(m,n)')
ax2.view_init(elev=25, azim=-60)
ax2.set_box_aspect((cutoff_display, cutoff_display, 0.8 * cutoff_display * 0.4))
ax2.set_xticks(np.arange(0, cutoff_display, 2))
ax2.set_yticks(np.arange(0, cutoff_display, 2))
ax2.set_zlim(0, z_c_vis.max() * 1.05)

plt.suptitle('Joint photon-number distribution P(m,n) — 3D histograms', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


