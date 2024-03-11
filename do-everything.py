#!/usr/bin/python

# Imports

import numpy as np
import pandas as pd

from pathlib import Path
from subprocess import run

import matplotlib as mpl
from matplotlib import pyplot as plt

# Define makesim function

filespersim = 1


def makesim(simdir, H):
    newsim = False
    H_str = "{:05.3f}".format(H)
    filenamebeg = H_str + "-"

    pattern = filenamebeg + "*.bin"
    while len(list(simdir.glob(pattern))) < filespersim:
        run(["./Program/bin/csg-perform", str(simdir), H_str])
        newsim = True

    pattern = filenamebeg + "obs.dat"
    if len(list(simdir.glob(pattern))) == 0 or newsim:
        run(["./Program/bin/csg-analyze", str(simdir), H_str])


# Define calcres function

L = 16

N = L**3


def calcres(simdir, H):
    H_str = "{:05.3f}".format(H)
    filename = H_str + "-obs.dat"
    filepath = simdir / filename
    data_dict = {}

    i_m = 0
    i_dr = 0
    with open(filepath) as file:
        for line in file:
            if line == "\n":
                i_dr = 0
                i_m += 1
            else:
                data_list = [float(num) for num in line.split()]
                data_dict[(i_m, i_dr)] = data_list
                i_dr += 1

    df = pd.DataFrame.from_dict(
        data_dict,
        orient="index",
        columns=[
            "m",
            "q(0)",
            "q_r(kx)",
            "q_i(kx)",
            "q_r(ky)",
            "q_i(ky)",
            "q_r(kz)",
            "q_i(kz)",
        ],
    )
    df.index = pd.MultiIndex.from_tuples(
        df.index,
        names=("i_m", "i_dr"),
    )

    df["|q(0)|^2"] = df["q(0)"] ** 2
    df["|q(kx)|^2"] = df["q_r(kx)"] ** 2 + df["q_i(kx)"] ** 2
    df["|q(ky)|^2"] = df["q_r(ky)"] ** 2 + df["q_i(ky)"] ** 2
    df["|q(kz)|^2"] = df["q_r(kz)"] ** 2 + df["q_i(kz)"] ** 2

    n_term = len(df.groupby(level="i_m")) // 2
    df.drop(index=range(n_term), level="i_m", inplace=True)
    df_mean = df.groupby(level="i_dr").mean().mean()

    M = df_mean["m"]
    chi_0 = df_mean["|q(0)|^2"]
    chi_k = (df_mean["|q(kx)|^2"] + df_mean["|q(ky)|^2"] + df_mean["|q(kz)|^2"]) / 3
    xi = np.sqrt(chi_0 / chi_k - 1) / (2 * np.sin(np.pi / L))

    #compute errors (sem) ...

    return [H, M, chi_0, chi_k, xi]


# Make simulations and calculate results

simdir = Path("Simulations")

n_points = 16

simdir.mkdir(exist_ok=True)

res_dict = {}
for i in range(n_points):
    H = 2 * (i + 1) / n_points
    makesim(simdir, H)
    res_dict[i] = calcres(simdir, H)
