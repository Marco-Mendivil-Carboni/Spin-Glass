#!/usr/bin/python

# Imports

import numpy as np

from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt

# Set matplotlib parameters

# mpl.use("pdf")

# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["font.family"] = "serif"

cm = 1 / 2.54
mpl.rcParams["figure.figsize"] = [24.00 * cm, 16.00 * cm]
mpl.rcParams["figure.constrained_layout.use"] = True

mpl.rcParams["legend.frameon"] = False

# Set constants and auxiliary variables

NREP = 24
NDIS = 1024

obs_dt = np.dtype(
    [
        ("e", "f4", (2,)),
        ("m", "f4", (2,)),
        ("q_0", "f4"),
        ("q_1_r", "f4", (3,)),
        ("q_1_i", "f4", (3,)),
    ]
)

beta = np.logspace(1, -3, num=NREP, base=2.0)

# Define analyze_obs function


def analyze_obs(dir, L, H, ax):
    sub_dir = dir / ("{:02}".format(L))
    file = sub_dir / ("{:06.4f}".format(H) + "-obs.bin")
    print("file_path : " + str(file))

    data = np.fromfile(file, dtype=obs_dt)
    data = data.reshape(-1, NDIS, NREP)
    data = np.delete(data, range(len(data) // 2), axis=0)
    print("data.shape : " + str(data.shape))

    e_avg = np.average(data["e"], axis=(0, 1))
    e = np.average(e_avg, axis=1)

    m_avg = np.average(data["m"], axis=(0, 1))
    m = np.average(m_avg, axis=1)

    q_0_var = np.var(data["q_0"], axis=(0, 1))
    q_1_r_var = np.var(data["q_1_r"], axis=(0, 1))
    q_1_i_var = np.var(data["q_1_i"], axis=(0, 1))
    q_1_var = q_1_r_var + q_1_i_var

    chi_0 = q_0_var
    chi_1 = np.average(q_1_var, axis=1)
    chi_frac = chi_0 / chi_1
    chi_frac = np.where(chi_frac > 1, chi_frac, 1)

    xi_L = np.sqrt(chi_frac - 1) / (2 * np.sin(np.pi / L))

    ax[0, 0].plot(beta, e)
    ax[0, 1].plot(beta, m)
    ax[1, 0].plot(beta, chi_0)
    ax[1, 1].plot(beta, xi_L / L)


# Analyze simulations

dir = Path("Simulations")

L = 16

n_H_val = 1

fig, ax = plt.subplots(2, 2)

analyze_obs(dir, L, 0, ax)

for i in range(n_H_val):
    H = (i + 1) / 16
    analyze_obs(dir, L, H, ax)

# View analysis

for iax in ax.ravel():
    iax.set_xscale("log")
    iax.set_xlabel("$\\beta$")

ax[0, 0].set_ylabel("$e$")
ax[0, 1].set_ylabel("$m$")
ax[1, 0].set_ylabel("$\\chi(0)$")
ax[1, 1].set_ylabel("$\\xi_L/L$")

plt.show()
