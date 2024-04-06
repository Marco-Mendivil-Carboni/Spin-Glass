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

color = ["#169f62", "#194bb2", "#a31cc5", "#d81e2c"]

sim_dir = Path("Simulations")
plots_dir = Path("Plots")

n_var = 4

n_H_v = 4
n_L_v = 3

# Define read_res function


def read_res(dir: Path, L: int, H: float) -> pd.DataFrame:
    sub_dir = dir / ("{:02}".format(L))
    filename = sub_dir / ("{:06.4f}".format(H) + "-res.dat")

    df_res = pd.read_csv(
        filename,
        delim_whitespace=True,
        comment="#",
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


# Define plot_var function


def plot_var(
    ax: plt.Axes, df_res: pd.DataFrame, var: str, color: str, label: str
) -> None:
    ax.plot(df_res["beta"], df_res[var + "_a"], color=color, label=label)
    ax.fill_between(
        df_res["beta"],
        df_res[var + "_a"] - df_res[var + "_e"],
        df_res[var + "_a"] + df_res[var + "_e"],
        alpha=0.50,
        color=color,
        linewidth=0.00,
    )


# Define init_plot function


def init_plot(axs: list[plt.Axes]) -> None:
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$\\beta$")

    axs[0].set_ylabel("$\\chi(0)$")
    axs[1].set_ylabel("$\\xi_L/L$")
    axs[2].set_ylabel("$e$")
    axs[3].set_ylabel("$m$")

    for ax in [axs[0], axs[1], axs[3]]:
        ax.axhline(
            y=0,
            alpha=0.25,
            color="#000000",
            linestyle="--",
            linewidth=1.00,
        )


# Define add_insets function


def add_insets(axs: list[plt.Axes]) -> list[plt.Axes]:
    iaxs: list[plt.Axes] = []
    iaxs.append(
        axs[0].inset_axes(
            [0.152, 0.184, 0.32, 0.36],
            xlim=(0.125, 0.475),
            ylim=(0.000, 0.002),
        )
    )
    iaxs.append(
        axs[1].inset_axes(
            [0.134, 0.242, 0.32, 0.36],
            xlim=(0.125, 0.275),
            ylim=(0.000, 0.030),
        )
    )
    iaxs.append(
        axs[2].inset_axes(
            [0.596, 0.544, 0.32, 0.36],
            xlim=(0.125, 0.126),
            ylim=(-0.38, -0.37),
        )
    )
    return iaxs


# Define add_legend function


def add_legend(axs: list[plt.Axes]) -> None:
    for ax in [axs[0], axs[1], axs[3]]:
        ax.legend()
    axs[2].legend(loc="lower left")


# Create plots

figs: list[plt.Figure] = []
axs: list[plt.Axes] = []
for _ in range(n_var):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)

# Make plots with L = 16 and H = 0.0

init_plot(axs)

L = 16
H = 0.0

df_res = read_res(sim_dir, L, H)

plot_var(axs[0], df_res, "chi_0_a", color[0], None)
plot_var(axs[1], df_res, "xi_LL_a", color[0], None)
plot_var(axs[2], df_res, "e_a", color[0], None)
plot_var(axs[3], df_res, "m_a", color[0], None)

figs[0].savefig(plots_dir / "0-chi_0.pdf")
figs[1].savefig(plots_dir / "0-xi_LL.pdf")
figs[2].savefig(plots_dir / "0-e.pdf")
figs[3].savefig(plots_dir / "0-m.pdf")

for ax in axs:
    ax.clear()

# Make plots with varying H

init_plot(axs)

iaxs = add_insets(axs)

L = 16
for i_H in range(n_H_v):
    H = i_H / 16

    df_res = read_res(sim_dir, L, H)
    label = "$H$ = {:06.4f}".format(H)

    plot_var(axs[0], df_res, "chi_0_a", color[i_H], label)
    plot_var(axs[1], df_res, "xi_LL_a", color[i_H], label)
    plot_var(axs[2], df_res, "e_a", color[i_H], label)
    plot_var(axs[3], df_res, "m_a", color[i_H], label)

    plot_var(iaxs[0], df_res, "chi_0_a", color[i_H], label)
    plot_var(iaxs[1], df_res, "xi_LL_a", color[i_H], label)
    plot_var(iaxs[2], df_res, "e_a", color[i_H], label)

add_legend(axs)

figs[0].savefig(plots_dir / "H-chi_0.pdf")
figs[1].savefig(plots_dir / "H-xi_LL.pdf")
figs[2].savefig(plots_dir / "H-e.pdf")
figs[3].savefig(plots_dir / "H-m.pdf")

axs[3].clear()

axs[3].set_xlabel("$\\beta$")
axs[3].set_ylabel("$m/H$")

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
for i_H in range(1, n_H_v):
    H = i_H / 16

    df_res = read_res(sim_dir, L, H)
    label = "$H$ = {:06.4f}".format(H)

    df_res["m_a_a"] /= H
    df_res["m_a_e"] /= H

    plot_var(axs[3], df_res, "m_a", color[i_H], label)

axs[3].legend()

figs[3].savefig(plots_dir / "H-mH.pdf")

for ax in axs:
    ax.clear()

# Make plots with varying L

init_plot(axs)

iaxs = add_insets(axs)

H = 0.0
for i_L in range(n_L_v):
    L = 16 - 2 * i_L

    df_res = read_res(sim_dir, L, H)
    label = "$L$ = {:02}".format(L)

    plot_var(axs[0], df_res, "chi_0_a", color[i_L], label)
    plot_var(axs[1], df_res, "xi_LL_a", color[i_L], label)
    plot_var(axs[2], df_res, "e_a", color[i_L], label)
    plot_var(axs[3], df_res, "m_a", color[i_L], label)

    plot_var(iaxs[0], df_res, "chi_0_a", color[i_L], label)
    plot_var(iaxs[1], df_res, "xi_LL_a", color[i_L], label)
    plot_var(iaxs[2], df_res, "e_a", color[i_L], label)

for ax in [axs[1], iaxs[1]]:
    ax.axvline(
        x=0.91,
        alpha=0.50,
        color=color[3],
        label="$\\beta$ = 0.91",
        linestyle="--",
        linewidth=1.00,
    )

add_legend(axs)

figs[0].savefig(plots_dir / "L-chi_0.pdf")
figs[1].savefig(plots_dir / "L-xi_LL.pdf")
figs[2].savefig(plots_dir / "L-e.pdf")
figs[3].savefig(plots_dir / "L-m.pdf")

iaxs[1].set_xlim(0.88, 0.94)
axs[1].set_xlim(0.20, 1.25)
iaxs[1].set_ylim(0.60, 0.70)
axs[1].set_ylim(0.00, 1.00)

figs[1].savefig(plots_dir / "L-xi_LL-z.pdf")
