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
mpl.rcParams["figure.figsize"] = [30.00 * cm, 20.00 * cm]
mpl.rcParams["figure.constrained_layout.use"] = True

mpl.rcParams["legend.frameon"] = False

# Set constants and auxiliary variables

NREP = 24
NDIS = 2048

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

col = ["#169f62", "#1880ac", "#221ab9", "#a31cc5"]

# Define analyze_obs function


def analyze_obs(dir: Path, L: int, H: float):
    sub_dir = dir / ("{:02}".format(L))
    file = sub_dir / ("{:06.4f}".format(H) + "-obs.bin")
    print("file : " + str(file))

    therm = file.stat().st_size // 2
    data = np.memmap(file, dtype=obs_dt, mode="r", offset=therm)
    data = data.reshape(-1, NDIS, NREP)
    print("data.shape : " + str(data.shape))

    e_a = np.average(data["e"], axis=(0, 3))
    e_a_a = np.average(e_a, axis=0)
    e_a_v = np.var(e_a, axis=0)
    e_a_e = np.sqrt(e_a_v / NDIS)

    m_a = np.average(data["m"], axis=(0, 3))
    m_a_a = np.average(m_a, axis=0)
    m_a_v = np.var(m_a, axis=0)
    m_a_e = np.sqrt(m_a_v / NDIS)

    chi_0_a = np.average(data["q_0"] ** 2, axis=0)
    chi_0_a_a = np.average(chi_0_a, axis=0)
    chi_0_a_v = np.var(chi_0_a, axis=0)
    chi_0_a_e = np.sqrt(chi_0_a_v / NDIS)

    chi_1_r_a = np.average(data["q_1_r"] ** 2, axis=0)
    chi_1_r_a_a = np.average(chi_1_r_a, axis=0)
    chi_1_r_a_v = np.var(chi_1_r_a, axis=0)
    chi_1_i_a = np.average(data["q_1_i"] ** 2, axis=0)
    chi_1_i_a_a = np.average(chi_1_i_a, axis=0)
    chi_1_i_a_v = np.var(chi_1_i_a, axis=0)
    chi_1_a_a = np.average(chi_1_r_a_a + chi_1_i_a_a, axis=1)
    chi_1_a_v = np.average(chi_1_r_a_v + chi_1_i_a_v, axis=1)

    chi_0_a_v_r = chi_0_a_v / chi_0_a_a**2
    chi_1_a_v_r = chi_1_a_v / chi_1_a_a**2

    chi_f_a_a = chi_0_a_a / chi_1_a_a
    chi_f_a_v = chi_f_a_a**2 * (chi_0_a_v_r + chi_1_a_v_r)

    # chi_f_a_a = np.where(chi_f_a_a > 1, chi_f_a_a, 1)

    xi_L_a_a = np.sqrt(chi_f_a_a - 1) / (2 * np.sin(np.pi / L))

    xi_L_a_v = (xi_L_a_a / (2 * (chi_f_a_a - 1))) ** 2 * chi_f_a_v
    xi_L_a_e = np.sqrt(xi_L_a_v / NDIS)

    # if H > 0.05: #remember to check this
    #     m = m / H
    return [
        e_a_a,
        e_a_e,
        m_a_a,
        m_a_e,
        chi_0_a_a,
        chi_0_a_e,
        xi_L_a_a / L,
        xi_L_a_e / L,
    ]


# Analyze simulations

dir = Path("Simulations")

fig, ax = plt.subplots(2, 2)

L = 16
for i in range(1):
    H = i / 16

    res = analyze_obs(dir, L, H)
    lab = "$H$ = {:06.4f}".format(H)

    ax[0, 0].plot(beta, res[0], color=col[i + 0], label=lab)
    ax[0, 0].fill_between(
        beta, res[0] - res[1], res[0] + res[1], alpha=0.25, color=col[i + 0]
    )
    ax[0, 1].plot(beta, res[2], color=col[i + 1], label=lab)
    ax[0, 1].fill_between(
        beta, res[2] - res[3], res[2] + res[3], alpha=0.25, color=col[i + 1]
    )
    ax[1, 0].plot(beta, res[4], color=col[i + 2], label=lab)
    ax[1, 0].fill_between(
        beta, res[4] - res[5], res[4] + res[5], alpha=0.25, color=col[i + 2]
    )
    ax[1, 1].plot(beta, res[6], color=col[i + 3], label=lab)
    ax[1, 1].fill_between(
        beta, res[6] - res[7], res[6] + res[7], alpha=0.25, color=col[i + 3]
    )

# H = 0.0
# for i in range(3):
#     L = 14 + 2 * i

#     res = analyze_obs(dir, L, H)
#     lab = "$L$ = {:02}".format(L)

#     ax[0, 0].plot(beta, res[0], color=col[i], linestyle="dashed", label=lab)
#     ax[0, 1].plot(beta, res[1], color=col[i], linestyle="dashed", label=lab)
#     ax[1, 0].plot(beta, res[2], color=col[i], linestyle="dashed", label=lab)
#     ax[1, 1].plot(beta, res[3], color=col[i], linestyle="dashed", label=lab)

# View analysis

for iax in ax.ravel():
    iax.set_xscale("log")
    iax.set_xlabel("$\\beta$")
    iax.legend()

ax[0, 0].set_ylabel("$e$")
ax[0, 1].set_ylabel("$m$")
ax[1, 0].set_ylabel("$\\chi(0)$")
ax[1, 1].set_ylabel("$\\xi_L/L$")

plt.show()
