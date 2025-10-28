# figD_noon_fringe_fixed.py
"""
Plot exact N-fold interference fringe for an ideal NOON state of chosen N.
"""
import numpy as np
import matplotlib.pyplot as plt

# choose N for the fringe 
N = 5
nphi = 721
phi_vals = np.linspace(0, 2*np.pi, nphi)

# For an ideal NOON state probability after recombining at a symmetric BS:
P_vals = np.cos(0.5 * N * phi_vals)**2

plt.figure(figsize=(8,4))
plt.plot(phi_vals, P_vals, lw=2)
plt.xlabel("Mach–Zehnder phase φ (radians)")
plt.ylabel(f"P(|{N},0⟩)")
plt.title(f"Ideal NOON: N = {N}  →  {N}-fold oscillations (0 → 2π)")
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])
plt.grid(True)
plt.tight_layout()
plt.show()
