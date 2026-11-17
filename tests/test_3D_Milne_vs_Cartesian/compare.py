#!/usr/bin/env python3
"""Milne against Cartesian on the common slice eta = 0 = z (3D blob test)."""
import sys
import numpy as np

NAMES = ["e", "rhob", "rhoq", "rhos", "u0", "u1", "u2", "u3"] + \
        ["W%d" % i for i in range(14)] + ["Pi"]
SHOW = {"e": "e", "u1": "u^x", "u2": "u^y", "u3": "u^z",
        "W4": "pi^xx", "W7": "pi^yy", "W9": "pi^zz", "Pi": "Pi"}


def load(path):
    with open(path, "rb") as f:
        nx, ny, neta, nf = np.frombuffer(f.read(16), dtype=np.int32)
        d = np.frombuffer(f.read(), dtype=np.float64)
    return d.reshape(nf, neta, ny, nx)   # index ix + nx*(iy + ny*ieta)


def slice0(a):
    return a[:, a.shape[1]//2, :, :]


def main(visc):
    """Prints the table; returns the number of failed checks: every field
    must converge like 1/tau0 (the ratio of the differences at tau0 = 20 and
    80 between 3 and 5) and be within 5e-2 of Cartesian at tau0 = 80."""
    rows = []
    for tau0 in (20, 40, 80):
        c = slice0(load(f"cart_{tau0}_v{visc}/state.bin"))
        m = slice0(load(f"milne_{tau0}_v{visc}/state.bin"))
        row = {}
        for k, lab in SHOW.items():
            i = NAMES.index(k)
            scale = np.abs(c[i]).max()
            row[lab] = np.abs(c[i] - m[i]).max()/scale if scale > 0 else 0.
        rows.append((tau0, row))
    labs = [l for l in SHOW.values() if any(r[l] > 0 for _, r in rows)]
    print("viscous" if visc else "ideal",
          "-- max |Milne - Cartesian| on z = 0, relative to the field's max")
    print("tau0 " + "".join("%11s" % l for l in labs))
    for tau0, r in rows:
        print("%4d " % tau0 + "".join("%11.2e" % r[l] for l in labs))
    r20, r80 = rows[0][1], rows[2][1]
    print("ratio 20/80 " + "".join("%11.2f" % (r20[l]/r80[l]) if r80[l] > 0
                                   else "%11s" % "-" for l in labs))
    failed = 0
    for l in labs:
        if l == "u^z":
            continue        # zero on z = 0 by symmetry: see check_uz
        ratio = r20[l]/r80[l]
        if not (3. < ratio < 5.) or r80[l] > 5e-2:
            print("FAILED:", l, "ratio", ratio, "at tau0 = 80:", r80[l])
            failed += 1
    return failed


def check_uz(visc):
    """u^z off the symmetric slice, at z = +-1 fm, where t and tau differ by
    z^2/(2 tau): the lab u^z of the Milne run, u^tau sinh(eta) + tau u^eta
    cosh(eta), against the Cartesian u^z; must converge like 1/tau0 too."""
    rel = []
    for tau0 in (20, 80):
        c = load(f"cart_{tau0}_v{visc}/state.bin")
        m = load(f"milne_{tau0}_v{visc}/state.bin")
        mid, k = c.shape[1]//2, 5
        deta = (16./tau0)/(c.shape[1] - 1)
        worst = 0.
        for s in (1, -1):
            ie, eta = mid + s*k, s*k*deta
            uz_m = m[4, ie]*np.sinh(eta) + m[7, ie]*np.cosh(eta)
            worst = max(worst, np.abs(uz_m - c[7, ie]).max()/np.abs(c[7, ie]).max())
        rel.append(worst)
    print("u^z at z = +-1 fm: %.2e (tau0 = 20), %.2e (tau0 = 80), ratio %.2f"
          % (rel[0], rel[1], rel[0]/rel[1]))
    return 0 if (3. < rel[0]/rel[1] < 5. and rel[1] < 5e-2) else 1


def check_t0():
    """The Cartesian run from t = 0 with shear and bulk viscosity: finite."""
    a = load("cart_0_v1/state.bin")
    ok = bool(np.isfinite(a).all())
    print("Cartesian from t = 0, shear and bulk: %s" % ("finite" if ok else "NOT finite"))
    return 0 if ok else 1


if __name__ == "__main__":
    failed = 0
    for v in (0, 1):
        failed += main(v) + check_uz(v)
    failed += check_t0()
    sys.exit(1 if failed else 0)
