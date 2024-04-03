#!/usr/bin/python

# Imports

import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

# Set matplotlib parameters

plt.rcParams["backend"] = "pdf"

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

cm = 1 / 2.54
plt.rcParams["figure.figsize"] = [12.00 * cm, 8.00 * cm]
plt.rcParams["figure.constrained_layout.use"] = True

plt.rcParams["legend.frameon"] = False

plt.rcParams["axes.formatter.limits"] = (-4, 4)

# Set constants and auxiliary variables

color = ["#169f62", "#1880ac", "#221ab9", "#a31cc5"]

# Define read_res function


def read_res(dir: Path, L: int, H: float) -> pd.DataFrame:
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
    var: str,
    color: str,
    label: str,
) -> None:
    ax.plot(
        df_res["beta"],
        df_res[var + "_a"],
        color=color,
        label=label,
    )
    ax.fill_between(
        df_res["beta"],
        df_res[var + "_a"] - df_res[var + "_e"],
        df_res[var + "_a"] + df_res[var + "_e"],
        alpha=0.50,
        color=color,
        linewidth=0.00,
    )


# Make plots

sim_dir = Path("Simulations")
plots_dir = Path("Plots")

n_figs = 4
n_H_val = 4
# n_L_val = 3

figs, axs = [], []
for _ in range(n_figs):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)

for ax in [axs[0], axs[1], axs[3]]:
    ax.axhline(
        y=0,
        alpha=0.25,
        color="black",
        linestyle="--",
        linewidth=1.00,
    )

L = 16
H = 0.0

df_res = read_res(sim_dir, L, H)

make_plot(axs[0], df_res, "chi_0_a", color[0], None)
make_plot(axs[1], df_res, "xi_LL_a", color[0], None)
make_plot(axs[2], df_res, "e_a", color[0], None)
make_plot(axs[3], df_res, "m_a", color[0], None)

for ax in axs:
    ax.set_xlabel("$\\beta$")
    ax.set_xscale("log", base=2)
axs[0].set_ylabel("$\\chi(0)$")
axs[1].set_ylabel("$\\xi_L/L$")
axs[2].set_ylabel("$e$")
axs[3].set_ylabel("$m$")

figs[0].savefig(plots_dir / "0/chi_0.pdf")
figs[1].savefig(plots_dir / "0/xi_LL.pdf")
figs[2].savefig(plots_dir / "0/e.pdf")
figs[3].savefig(plots_dir / "0/m.pdf")

for ax in axs:
    ax.clear()

for ax in [axs[0], axs[1], axs[3]]:
    ax.axhline(
        y=0,
        alpha=0.25,
        color="black",
        linestyle="--",
        linewidth=1.00,
    )

iax_1 = axs[1].inset_axes(
    [0.125, 0.225, 0.30, 0.35],
    xlim=(0.125, 0.275),
    ylim=(0.000, 0.030),
)

L = 16
for i_H in range(n_H_val):
    H = i_H / 16

    df_res = read_res(sim_dir, L, H)
    label = "$H$ = {:06.4f}".format(H)

    make_plot(axs[0], df_res, "chi_0_a", color[i_H], label)
    make_plot(axs[1], df_res, "xi_LL_a", color[i_H], label)
    make_plot(axs[2], df_res, "e_a", color[i_H], label)
    make_plot(axs[3], df_res, "m_a", color[i_H], label)

    make_plot(iax_1, df_res, "xi_LL_a", color[i_H], label)

for ax in axs:
    ax.set_xlabel("$\\beta$")
    ax.set_xscale("log", base=2)
    ax.legend()
axs[0].set_ylabel("$\\chi(0)$")
axs[1].set_ylabel("$\\xi_L/L$")
axs[2].set_ylabel("$e$")
axs[3].set_ylabel("$m$")

figs[0].savefig(plots_dir / "H/chi_0.pdf")
figs[1].savefig(plots_dir / "H/xi_LL.pdf")
figs[2].savefig(plots_dir / "H/e.pdf")
figs[3].savefig(plots_dir / "H/m.pdf")

axs[3].clear()

axs[3].axline(
    (0.00, 0.00),
    (0.80, 0.80),
    alpha=0.50,
    color=color[0],
    label="$m/H$ = $\\beta$",
    linestyle="--",
    linewidth=1.00,
)

L = 16
for i_H in range(1, n_H_val):
    H = i_H / 16

    df_res = read_res(sim_dir, L, H)
    label = "$H$ = {:06.4f}".format(H)

    df_res["m_a_a"] /= H
    df_res["m_a_e"] /= H

    make_plot(axs[3], df_res, "m_a", color[i_H], label)

axs[3].set_xlabel("$\\beta$")
axs[3].legend()
axs[3].set_ylabel("$m/H$")

figs[3].savefig(plots_dir / "H/mH.pdf")
