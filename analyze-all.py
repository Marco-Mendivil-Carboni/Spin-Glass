#!/usr/bin/python

# Imports

import numpy as np

from pathlib import Path

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

# Define analyze_obs function


def analyze_obs(dir: Path, L: int, H: float) -> None:
    sub_dir = dir / ("{:02}".format(L))
    filename = sub_dir / ("{:06.4f}".format(H) + "-obs.bin")
    print("filename : " + str(filename))

    therm_b = filename.stat().st_size // 2
    data = np.memmap(filename, dtype=obs_dt, mode="r", offset=therm_b)
    data = data.reshape(-1, NDIS, NREP)
    print("data.shape[0] : " + str(data.shape[0]))

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

    xi_LL_a_a = (np.sqrt(chi_f_a_a - 1) / (2 * np.sin(np.pi / L))) / L

    xi_LL_a_v = (xi_LL_a_a / (2 * (chi_f_a_a - 1))) ** 2 * chi_f_a_v
    xi_LL_a_e = np.sqrt(xi_LL_a_v / NDIS)

    e_a = np.average(data["e"], axis=(0, 3))
    e_a_a = np.average(e_a, axis=0)
    e_a_v = np.var(e_a, axis=0)
    e_a_e = np.sqrt(e_a_v / NDIS)

    m_a = np.average(data["m"], axis=(0, 3))
    m_a_a = np.average(m_a, axis=0)
    m_a_v = np.var(m_a, axis=0)
    m_a_e = np.sqrt(m_a_v / NDIS)

    filename = sub_dir / ("{:06.4f}".format(H) + "-res.dat")
    file = open(filename, "w")
    for i_r in range(NREP):
        file.write("{:16.12f}".format(beta[i_r]))
        file.write("{:16.12f}".format(chi_0_a_a[i_r]))
        file.write("{:16.12f}".format(chi_0_a_e[i_r]))
        file.write("{:16.12f}".format(xi_LL_a_a[i_r]))
        file.write("{:16.12f}".format(xi_LL_a_e[i_r]))
        file.write("{:16.12f}".format(e_a_a[i_r]))
        file.write("{:16.12f}".format(e_a_e[i_r]))
        file.write("{:16.12f}".format(m_a_a[i_r]))
        file.write("{:16.12f}".format(m_a_e[i_r]))
        file.write("\n")
    file.close()


# Analyze simulations

dir = Path("Simulations")

analyze_obs(dir, 16, 0.0000)
analyze_obs(dir, 16, 0.0625)
# analyze_obs(dir, 16, 0.1250)
# analyze_obs(dir, 16, 0.1875)

# analyze_obs(dir, 14, 0.0000)
# analyze_obs(dir, 18, 0.0000)
