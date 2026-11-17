#!/usr/bin/env python3

import sys

import numpy as np

# Sums over the 401 cells of |x|, |e|, |v|, |theta|, |pi|, |Pi| at t = 4 fm/c.
# The x and y tests evolve pi^xx (pi^yy) directly; in the z test MUSIC
# rebuilds pi^zz from the tracelessness of the evolved transverse components,
# so the z result differs from x at the shock front (0.07 GeV/fm^3 at most,
# L1 1.7e-4 of the total) and has its own reference.  Regenerated with the
# Cartesian guards (cartesian_guards / trackB); the earlier references
# (3108.52...) were made with the Milne terms of the viscous part active.
TARGETS = {
    "x": [1.0050000000000000e+03, 3.1206265122199998e+03,
          9.5883261830525683e+01, 4.4770896450402638e+01,
          4.2522155836702254e+01, 0.0000000000000000e+00],
    "z": [1.0050000000000000e+03, 3.1206353397200028e+03,
          9.5809764568683335e+01, 4.5147159604964614e+01,
          4.2467326967697446e+01, 0.0000000000000000e+00],
}
TARGETS["y"] = TARGETS["x"]
direction = sys.argv[1] if len(sys.argv) > 1 else "x"
TargetSum = TARGETS[direction]

data = np.loadtxt("1+1D_RiemannTest_tau_4.dat")
ResSum = np.sum(abs(data), axis=0)

Nfailed = 0
for i, val_i in enumerate(TargetSum):
    if abs(val_i - ResSum[i]) > 1e-8:
        print("Diff: ", val_i, ResSum[i])
        Nfailed += 1
