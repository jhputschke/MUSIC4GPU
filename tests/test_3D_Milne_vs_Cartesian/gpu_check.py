#!/usr/bin/env python3
"""Checks of RunGPUTest.sh: Cartesian coordinates in the CUDA kernels.

The limits sit between the correct kernels and planted errors, measured on
the GB10 (Christoffel terms left in sigma^{mu nu}; the u^tau/tau piece left
in theta; -u^tau Pi/tau left in the bulk equation):

  check                          correct   sigma    theta    -u Pi/tau
  x <-> z, 3 fm/c: e             3.2e-4    1.5e-2   1.6e-1   3.3e-4
                   pi^xy/pi^yz   3.1e-3    7.1e-1   6.1e-1   3.1e-3
                   pi^xx/pi^zz   2.7e-2    6.2e-1   1.5      2.0e-2
  GPU vs CPU, 10 steps: e        3.3e-4    3.3e-4   2.4e-4   3.3e-4
                   u^tau         1.7e-6    1.8e-6   1.0e-3   1.7e-6
                   pi^xx         1.0e-2    1.1e-2   4.3e+1   1.0e-2
                   Pi            1.1e-2    1.1e-2   1.4e+1   6.9e-2

(GPU vs CPU on cells with e > 5 % of the maximum; float32 against float64.)
"""
import sys
import numpy as np

U, W, PI = 4, 8, 22   # Fields order: e, rhob, rhoq, rhos, u[4], Wmunu[14], pi_b


def load(path):
    with open(path, "rb") as f:
        nx, ny, neta, nf = np.frombuffer(f.read(16), dtype=np.int32)
        d = np.frombuffer(f.read(), dtype=np.float64)
    return d.reshape(nf, neta, ny, nx)      # [field, z, y, x]


def rel(a, b):
    s = np.abs(b).max()
    return np.abs(a - b).max() / s if s > 0 else 0.


def check(label, value, limit):
    ok = value < limit
    print("%-34s %.1e (< %.0e) %s" % (label, value, limit, "" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    for r in ("gpu_0", "gpu_20", "gpu10_20"):
        log = open(f"{r}/run.log").read()
        if "using CPU" in log or "CUDA device" not in log:
            print(f"{r}: did not run on the GPU (CUDA build and device needed)")
            return 1
    failed = 0
    g0, g20 = load("gpu_0/state.bin"), load("gpu_20/state.bin")
    if not np.array_equal(g0, g20):
        print("GPU t0 = 0 and t0 = 20 differ: a kernel depends on t   FAILED")
        failed += 1
    else:
        print("GPU t0 = 0 and t0 = 20 bit-identical")

    sw = lambda f: np.transpose(f, (2, 1, 0))    # x <-> z
    failed += check("x <-> z, 3 fm/c: e", rel(g20[0], sw(g20[0])), 2e-3)
    failed += check("x <-> z, 3 fm/c: pi^xy/pi^yz",
                    rel(g20[W + 5], sw(g20[W + 8])), 2e-2)
    failed += check("x <-> z, 3 fm/c: pi^xx/pi^zz",
                    rel(g20[W + 4], sw(g20[W + 9])), 1e-1)

    g, c = load("gpu10_20/state.bin"), load("cpu10_20/state.bin")
    core = c[0] > 0.05 * c[0].max()
    for label, i, lim in (("e", 0, 1e-3), ("u^tau", U, 1e-5),
                          ("pi^xx", W + 4, 5e-2), ("Pi", PI, 5e-2)):
        failed += check("GPU vs CPU, 10 steps: " + label,
                        rel(g[i][core], c[i][core]), lim)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
