#!/usr/bin/python

# Imports

import numpy as np

from pathlib import Path

# import matplotlib as mpl
from matplotlib import pyplot as plt


# Define calcres function

L = 16
N = L**3

NREP = 24
NDIS = 256

obs_dt = np.dtype(
    [
        ("e", "f4", (2,)),
        ("m", "f4", (2,)),
        ("q_0", "f4"),
        ("q_r", "f4", (3,)),
        ("q_i", "f4", (3,)),
    ]
)


def calc_res(sim_dir, H):
    file_path = sim_dir / ("{:05.3f}".format(H) + "-obs.bin")

    data = np.fromfile(file_path, dtype=obs_dt)
    data = data.reshape(-1, NDIS, NREP)
    data = np.delete(data, range(len(data) // 4), axis=0)
    print(data.shape)

    something = (data["e"].mean(axis=0)).mean(axis=2)
    print(something.shape)
    something = something.mean(axis=0)
    plt.plot(something)
    plt.show()
    # ...

    # for line in file:
    #     i_m = idx // (NDIS * NREP)
    #     i_d = (idx % (NDIS * NREP)) // NREP
    #     i_r = (idx % (NDIS * NREP)) % NREP
    #     idx += 1

    # compute errors (sem) ...

    return [H]


# Make simulations and calculate results

simdir = Path("Simulations")

n_points = 12

# for i in range(n_points):
#     H = (i + 1) / n_points
#     ... = calc_res(simdir, H)

H = 0.0
print(calc_res(simdir, H))
