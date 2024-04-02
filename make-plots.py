#!/usr/bin/python

# Imports

import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

# Set matplotlib parameters

# plt.rcParams["backend"] = "pdf"

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"

cm = 1 / 2.54
plt.rcParams["figure.figsize"] = [30.00 * cm, 20.00 * cm]
plt.rcParams["figure.constrained_layout.use"] = True

plt.rcParams["legend.frameon"] = False

# Set constants and auxiliary variables

color = ["#169f62", "#1880ac", "#221ab9", "#a31cc5"]

# Define read_res function


def read_res(
    dir: Path,
    L: int,
    H: float,
) -> pd.DataFrame:
    sub_dir = dir / ("{:02}".format(L))
    filename = sub_dir / ("{:06.4f}".format(H) + "-res.dat")

    df_res = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
    )
    df_res.columns = [
        "beta",
        "chi_0_a_a",
        "chi_0_a_e",
        "xi_LL_a_a",
        "xi_LL_a_e",
        "e_a_a",
        "e_a_e",
        "m_a_a",
        "m_a_e",
    ]

    return df_res


# Define make_plot function


def make_plot(
    ax: plt.Axes,
    df_res: pd.DataFrame,
    column_a: str,
    column_e: str,
    color: str,
    label: str,
) -> None:
    ax.plot(
        df_res["beta"],
        df_res[column_a],
        color=color,
        label=label,
    )
    ax.fill_between(
        df_res["beta"],
        df_res[column_a] - df_res[column_e],
        df_res[column_a] + df_res[column_e],
        color=color,
        alpha=0.50,
        linewidth=0.0,
    )


# Make plots

# if H > 0.05: #remember to check this
#     m = m / H

dir = Path("Simulations")

fig, axs = plt.subplots(2, 2)

L = 16
for i in range(2):
    H = i / 16
    df_res = read_res(dir, L, H)
    label = "$H$ = {:06.4f}".format(H)

    make_plot(axs[0, 0], df_res, "chi_0_a_a", "chi_0_a_e", color[i], label)
    make_plot(axs[0, 1], df_res, "xi_LL_a_a", "xi_LL_a_e", color[i], label)
    make_plot(axs[1, 0], df_res, "e_a_a", "e_a_e", color[i], label)
    make_plot(axs[1, 1], df_res, "m_a_a", "m_a_e", color[i], label)

# View analysis

for ax in axs.ravel():
    ax.set_xscale("log")
    ax.set_xlabel("$\\beta$")
    ax.legend()

axs[0, 0].set_ylabel("$\\chi(0)$")
axs[0, 1].set_ylabel("$\\xi_L/L$")
axs[1, 0].set_ylabel("$e$")
axs[1, 1].set_ylabel("$m$")

plt.show()
