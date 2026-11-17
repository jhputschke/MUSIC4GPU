# Cartesian coordinates (CoorType 1)

Status: November 2026, branch `trackB` (`bqs_gpu` + `cartesian_guards`).
Track B of the "Cartesian MUSIC on GPU" plan: a Cartesian hydro for running
next to SMASH at low beam energies. This file covers step B1, the CPU
reference. The GPU kernels are Milne only: `gpu_features_supported()` sends
`CoorType 1` to the CPU path until step B2.

## What changed (B1)

In Cartesian coordinates MUSIC replaces tau by 1 in the conserved variables
and the spacing; the Christoffel terms of Milne coordinates must vanish.
Places that still applied Milne geometry with `CoorType 1`:

| Where | Term | Fixed in |
|---|---|---|
| `Diss::MakeWSource` | the z flux of W^{z0}, W^{zz} (and of the bulk part) set aside for the cosh/sinh form of the eta derivative, added back only for Milne: the z derivative of the viscous stress never reached the energy and z-momentum equations | `cartesian_guards` |
| `Diss::MakeWSource` | "+ W", "+ Pi" from partial_tau (tau W^{tau alpha}) | `cartesian_guards` |
| `Diss::Make_uWRHS` | the Christoffel terms ~ u^3/tau; - u^tau W/tau and its bulk counterpart (a division by zero at t = 0) | `cartesian_guards` |
| `Diss::Make_uqSource` (baryon diffusion) | - q u^tau/tau, Delta^{a tau} u^eta q^eta/tau, Delta^{a eta} u^eta q^tau/tau | `trackB` |
| `U_derivative::calculate_thermal_shear_tensor`, `calculate_T_vorticity` (output only) | the Christoffel terms | `trackB` |

Milne runs are unchanged: every deterministic output file is byte-identical
to `bqs_gpu` (Gubser flow ideal, with baryon diffusion, and viscous with
shear, bulk, vorticity output and freeze-out; MUSIC_FORCE_CPU). The
eccentricity and momentum-anisotropy files differ from run to run already.

## The one-cell offset of the z tests

The z shock tube started one cell off the x and y tubes. Not physics: the
test profiles put the discontinuity at coordinate 0, the centre of a cell,
and compared with `> 0`. On aarch64 GCC fuses `delta_eta*ieta - eta_size/2`
into a multiply-add in the eta loop, which gives +2.8e-16 at the centre (the
x and y loops give 0). `initial_1p1D_Riemann` and `initial_1p1D_Diffusion`
now compare with a 1e-9 tolerance.

## Tests

| Test | Result |
|---|---|
| `test_1+1DRiemannTest_Cartesian`, ideal | x, y, z identical; all equal the stored reference |
| `test_1+1DRiemannTest_Cartesian`, viscous (eta/s = 0.1) | x, y identical; z within 0.07 GeV/fm^3 of x, only at the shock front (L1 1.7e-4 of the total); new references, z its own |
| `test_1+1DDiffusionTest_Cartesian` | x, y, z identical; all equal the stored reference |
| `test_3D_Milne_vs_Cartesian` (new) | the same 3D state in both coordinate systems: differences fall like 1/tau0 in every field; Cartesian from t = 0 with shear and bulk stays finite |

**Viscous z against x.** MUSIC evolves W^{xx}, W^{xy}, W^{xz}, W^{yy},
W^{yz} (and Pi) and rebuilds W^{zz} from the tracelessness of these (the
Milne layout, where eta is the special direction). In the x test pi^{xx} is
transported directly; in the z test pi^{zz} comes from the transported
W^{xx}, W^{yy}, which carry a factor 1/(u^0)^2 that varies from cell to
cell, so their limited fluxes are not proportional to a direct pi^{zz} flux.
The difference sits at the shock front and falls slowly with resolution (L1
1.32e-2 at 401 cells, 1.10e-2 at 801), as at any discontinuity. Making z
identical to x would mean transporting W^{zz} in Cartesian mode, a scheme
change left out of B1.

**3D Milne against Cartesian** (`Initial_profile 5`): a Gaussian blob at
rest in the lab frame (e0 10 GeV/fm^3, widths 1, 1.5, 1 fm, 81^3 cells of
0.2 fm, ideal gas), defined in physical coordinates, so that both runs start
from the same state: in Milne the cell at (tau0, eta) sits at z = tau0
sinh(eta) and the lab rest frame is u^tau = cosh(eta), tau u^eta =
-sinh(eta). The starting surfaces (t = t0, tau = tau0) differ by about
z^2/(2 tau0), so the difference after 3 fm/c must fall like 1/tau0. On the
common slice z = 0 = eta (largest difference relative to the field's
maximum):

| tau0 [fm] | e | u^x | u^y | pi^xx | pi^yy | pi^zz | Pi |
|---|---|---|---|---|---|---|---|
| 20 | 5.4e-2 | 5.5e-3 | 5.0e-3 | 5.6e-2 | 3.7e-2 | 1.8e-1 | 3.0e-2 |
| 40 | 2.7e-2 | 2.8e-3 | 2.6e-3 | 2.9e-2 | 1.9e-2 | 8.8e-2 | 1.5e-2 |
| 80 | 1.3e-2 | 1.4e-3 | 1.3e-3 | 1.5e-2 | 9.4e-3 | 4.4e-2 | 7.5e-3 |

(viscous: eta/s 0.08, bulk, second-order terms; the ideal runs converge the
same way.) u^z is zero on z = 0 in both (1e-13); at z = +-1 fm its
difference falls from 0.15 (tau0 = 20) to 0.036 (80).

## Not done in B1

- Upstream confirmation of the guards (MUSIC-fluid/MUSIC branch `XSCAPE`,
  `cee9460`, where the same lines sit at the same line numbers).
- `Make_uWRHS_geom` serves the GPU path only (B2).
- Transporting W^{zz} directly in Cartesian mode (see above).
