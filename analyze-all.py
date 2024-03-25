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

L = 16
N = L**3

NREP = 24
NDIS = 256

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


def analyze_obs(sim_dir, H, ax):
    file_path = sim_dir / ("{:06.4f}".format(H) + "-obs.bin")

    data = np.fromfile(file_path, dtype=obs_dt)
    data = data.reshape(-1, NDIS, NREP)
    data = np.delete(data, range(len(data) // 4), axis=0)
    print(data.shape)
    e_avg = np.average(np.average(data["e"], axis=0), axis=2)
    e_std = np.average(np.std(data["e"], axis=0), axis=2)
    e_avg_avg = np.average(e_avg, axis=0)
    e_avg_std = np.std(e_avg, axis=0)
    e_std_avg = np.average(e_std, axis=0)

    m_avg = np.average(np.average(data["m"], axis=0), axis=2)
    m_std = np.average(np.std(data["m"], axis=0), axis=2)
    m_avg_avg = np.average(m_avg, axis=0)
    m_avg_std = np.std(m_avg, axis=0)
    m_std_avg = np.average(m_std, axis=0)

    q_0_var = np.var(data["q_0"], axis=(0, 1))
    q_1_r_var = np.var(data["q_1_r"], axis=(0, 1))
    q_1_i_var = np.var(data["q_1_i"], axis=(0, 1))
    q_1_var = q_1_r_var + q_1_i_var

    chi_0_avg = q_0_var
    chi_1_avg = np.average(q_1_var, axis=1)

    chi_frac = chi_0_avg / chi_1_avg
    chi_frac = np.where(chi_frac > 1, chi_frac, 1)

    xi_L_avg = np.sqrt(chi_frac - 1) / (2 * np.sin(np.pi / L))

    ax[0, 0].plot(beta, e_avg_avg)
    ax[0, 1].plot(beta, e_std_avg)
    ax[0, 2].plot(beta, e_avg_std)
    ax[1, 0].plot(beta, m_avg_avg)
    ax[1, 1].plot(beta, m_std_avg)
    ax[1, 2].plot(beta, m_avg_std)
    ax[2, 0].plot(beta, chi_0_avg)
    ax[2, 1].plot(beta, chi_1_avg)
    ax[2, 2].plot(beta, xi_L_avg / L)


# Analyze simulations

simdir = Path("Simulations")

fig, ax = plt.subplots(3, 3)

for a in ax.ravel():
    a.set_xscale("log")

analyze_obs(simdir, 0.0, ax)

n_H_val = 0

for i in range(n_H_val):
    H = (2**i) / 16
    analyze_obs(simdir, H, ax)

# View analysis

plt.show()
