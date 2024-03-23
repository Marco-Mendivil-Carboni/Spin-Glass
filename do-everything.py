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

# cm = 1 / 2.54
# mpl.rcParams["figure.figsize"] = [12.00 * cm, 8.00 * cm]
mpl.rcParams["figure.constrained_layout.use"] = True

mpl.rcParams["legend.frameon"] = False

# Define analyze_obs function

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


def analyze_obs(sim_dir, H):
    file_path = sim_dir / ("{:05.3f}".format(H) + "-obs.bin")

    data = np.fromfile(file_path, dtype=obs_dt)
    data = data.reshape(-1, NDIS, NREP)
    data = np.delete(data, range(len(data) // 4), axis=0)

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

    q_0_2_avg = np.average(data["q_0"] ** 2, axis=0)
    q_0_2_avg_avg = np.average(q_0_2_avg, axis=0)

    q_0_avg = np.average(data["q_0"], axis=0)
    q_0_avg_avg = np.average(q_0_avg, axis=0)

    q_2_avg = np.average(data["q_1_r"] ** 2 + data["q_1_i"] ** 2, axis=0)
    q_2_avg_avg = np.average(q_2_avg, axis=0)

    q_r_avg = np.average(data["q_1_r"], axis=0)
    q_r_avg_avg = np.average(q_r_avg, axis=0)

    q_i_avg = np.average(data["q_1_i"], axis=0)
    q_i_avg_avg = np.average(q_i_avg, axis=0)

    chi_0_avg = q_0_2_avg_avg - q_0_avg_avg**2

    chi_avg = np.average(
        q_2_avg_avg - (q_r_avg_avg**2 + q_i_avg_avg**2),
        axis=1,
    )

    xi_avg = np.sqrt(np.maximum(chi_0_avg / chi_avg - 1, np.zeros(NREP))) / (
        2 * np.sin(np.pi / L)
    )

    plt.plot(beta, e_avg_avg)
    plt.plot(beta, e_std_avg)
    plt.plot(beta, e_avg_std)
    plt.plot(beta, m_avg_avg)
    plt.plot(beta, m_std_avg)
    plt.plot(beta, m_avg_std)
    plt.plot(beta, chi_0_avg)
    plt.plot(beta, chi_avg)
    plt.plot(beta, xi_avg / L)

    plt.xscale("log")
    plt.show()

    return [H]


# Analyze observables

simdir = Path("Simulations")

n_points = 1

for i in range(n_points):
    H = (i + 0) / 8
    print(analyze_obs(simdir, H))

# Make plots
