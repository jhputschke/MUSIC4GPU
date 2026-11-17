# Cartesian coordinates (CoorType 1)

Status: November 2026, branch `trackB` (`bqs_gpu` + `cartesian_guards` +
`active_region`). Track B of the "Cartesian MUSIC on GPU" plan: a Cartesian
hydro for running next to SMASH at low beam energies. This file covers step
B1, the CPU reference, step B2, Cartesian coordinates in the CUDA
kernels, and the drain of particlized matter for X-SCAPE's dynamic
fluidization. Metal stays Milne only: there `gpu_features_supported()` sends
`CoorType 1` to the CPU path.

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

## Cartesian coordinates in the CUDA kernels (B2)

One flag, `MUSICGridParams::cartesian` (in the CUDA-only tail of the
struct), and the host settings of `Advance::make_gpu_params` do most of it,
as `tauRkFactor`/`tauNextFactor` do on the CPU:

| Host setting (CoorType 1) | Effect in the kernels |
|---|---|
| `tau = tau_orig = 1` | conserved variables, fluxes, reconstruction and the RK combination without tau weights; the z spacing `delta_eta*tau` becomes `delta_eta`; `1/tau` in the z signal speed becomes 1 |
| `cosh_deta = 1/delta_eta`, `sinh_deta = 0` | the cosh/sinh form of the eta flux of T^{0 alpha}, T^{3 alpha}, W, Pi becomes the plain flux difference |

and the kernels drop the terms that are geometric, not a tau weight:

| Kernel | Term dropped where `cartesian` is set |
|---|---|
| `gpu_make_w_source`, `_tiled` | "+ W", "+ Pi" from partial_tau (tau W^{tau alpha}) |
| `gpu_finalize_ideal`, `_eos4d` | tau_next = 1 (tauNextFactor) |
| `gpu_make_du` | u^tau/tau in theta; the Christoffel terms of sigma^{mu nu} |
| `gpu_uWRHS_geom` (both `w_full` kernels) | the Christoffel terms ~ u^3/tau; - u^tau W/tau |
| `gpu_first_rk_step_w_full`, `_eos4d` | - u^tau Pi/tau |
| `gpu_first_rk_step_w_full_eos4d` (baryon diffusion) | - q u^tau/tau and the Delta^{a tau}, Delta^{a eta} terms |

`gpu_features_supported()` lets `CoorType 1` through on CUDA. The source
pre-fill already used `tauRkFactor`, so jet and droplet sources need nothing.

**Milne stays bit-identical**, and that needs care. The first version put
the Milne and Cartesian expressions in the two arms of a branch; the Milne
output then changed in the last bits (seed-42 SMASH-IC event: 633 instead of
638 hadrons, the freeze-out surface identical to its printed precision).
Arithmetic shared by both arms (e.g. `theta*pi_c`) is computed once, in a
block of its own, and the Milne addition that used to be fused with it into
one FMA is not any more. Now every Milne expression is computed exactly as
before, unconditionally, and the Cartesian value overrides it afterwards,
written with the `_rn` intrinsics (`__fmul_rn`, `__fadd_rn`, ...), which the
compiler never contracts and never merges with ordinary arithmetic. Checked:
standalone Milne runs with the 1D EOS, with NEOS-4D, and with NEOS-4D and
baryon diffusion, byte-identical state dumps against `c0841df`; the staged
Milne SMASH-IC event (seed 42) byte-identical in its hadrons to the run
before B2. The rule for later edits: no Cartesian arithmetic in a branch
around Milne arithmetic; override afterwards, with intrinsics.

**Tests.** `tests/test_3D_Milne_vs_Cartesian/RunGPUTest.sh` (CUDA builds,
half a minute): the viscous blob of the 3D test on the GPU. The limits sit
between the correct kernels and three planted errors (the table in
`gpu_check.py`):

| Check | Correct | Christoffel terms left in sigma | u^tau/tau left in theta | - u^tau Pi/tau left |
|---|---|---|---|---|
| GPU t0 = 0 against t0 = 20 (nothing may depend on t) | bit-identical | bit-identical | bit-identical | bit-identical |
| x <-> z mirror symmetry after 3 fm/c (sigma_x = sigma_z): e | 3.2e-4 | 1.5e-2 | 1.6e-1 | 3.3e-4 |
| same, pi^xy against pi^yz | 3.1e-3 | 7.1e-1 | 6.1e-1 | 3.1e-3 |
| GPU against CPU after 10 steps, core: u^tau | 1.7e-6 | 1.8e-6 | 1.0e-3 | 1.7e-6 |
| same, Pi | 1.1e-2 | 1.1e-2 | 1.4e+1 | 6.9e-2 |

The x <-> z residual of the correct kernels is the W^{zz} rebuild (the CPU
shows the same, 2.5e-2 in pi^xx/pi^zz). GPU against CPU in Cartesian
coordinates differs as much as in Milne coordinates. Blob after 3 fm/c: e
2.7e-2 (Milne 2.5e-2) of the maximum, in the expanding shell. Viscous shock
tube, x: e 0.54 % (Milne at tau0 = 100: 0.54 %), theta 2.8 % (5.1 %), pi
2.7 % (2.9 %); x and z show the same offset. So the viscous shock tubes do not
reproduce their CPU reference sums on the GPU (`TestOutputVisFiles.py` asks
for 1e-8); they are CPU tests (`MUSIC_FORCE_CPU=1`).

**Speed.** Blob, 81^3 cells, 150 steps, shear + bulk + second order: GPU
(GB10) 3.75 s, CPU (20 threads) 31 s.

Not on the GPU, in either coordinate system: ideal hydro
(`Viscosity_Flag_Yes_1_No_0 0`) and the vorticity term of the shear
equation (`Include_vorticity_terms 1` is ignored by the kernels).

## Draining particlized matter (`drain_particlized`)

X-SCAPE's dynamic fluidization runs this hydro next to SMASH, fed by source
terms all the time, and particlizes every freeze-out check's surface at once
(iSS), handing the hadrons back to SMASH. MUSIC never removes particlized
matter: the fluid below the freeze-out density stays on the grid. Next to a
growing dense region (sources, flow) it crosses the surface again, inward
(u.dsigma < 0), and later outward a second time. iSS drops the inward
elements, so it is particlized twice. Au+Au at 7.7 GeV, b 0-3 fm, 3 events:
the outflow iSS samples is 5-14 % above what was fluidized (B 327 of 299,
331 of 315, 316 of 278), final states 107 % in E, B 413-421 of 394.

`drain_particlized 1` (X-SCAPE: `<BDM><drain_particlized_fluid>`, default 1
with dynamic fluidization; Cartesian only) takes that matter off:

- During the surface search each element with u.dsigma > 0 adds its
  T^{0 mu} dsigma_mu to the corners of its cube below the freeze-out density
  in the current step, shared by their energy
  (`FindFreezeOutSurface_Cornelius_XY`, `drain_share_`).
- After the search each such cell keeps the fraction 1 - share/E of e, n,
  W^{mu nu} and Pi at its u (`Evolve::drain_particlized_cells`), in the
  current and the previous step: a kernel when the GPU holds the state
  (`gpu_drain_cells`), the host arenas otherwise.
- A drained cell keeps at least `drain_particlized_floor` (1e-3 GeV/fm^3).
  Emptying cells to vacuum next to the surface made float vacuum interfaces
  there; their GPU cells blew up and reached the surface by interpolation
  (Pi ~ 1e14 GeV/fm^3 on surface elements; on the CPU the same run is clean).
- Matter that never reached the freeze-out density (the tails of the
  sources) is not touched; when the dense region reaches it, it is
  particlized once.

`Evolve::drained_totals` and `fluid_totals` (`MUSIC::get_drained_totals`,
`get_fluid_totals`) give the books; X-SCAPE prints them per event with iSS's
surface fluxes. The same 3 events with the drain:

| per event | B | E [GeV] |
|---|---|---|
| fluidized (SMASH -> MUSIC) | 295, 310, 291 | 1010, 1087, 1034 |
| outflow iSS gets (u.dsigma > 0) | 214, 228, 213 | 730, 792, 745 |
| drained | 216, 231, 215 | 707, 770, 726 |
| left on the grid at the end | 62, 61, 60 | 232, 240, 237 |
| through the grid edges (the rest) | 17, 18, 16 | 71, 77, 71 |
| particlized (iSS, one sample) | 191, 209, 191 | 683, 807, 747 |

Drained and outflow agree to 1-3 %: nothing is particlized twice. The final
states are 80-83 % in E, B 290-294 of 394: what the drain leaves visible is
lost matter (the fluid left on the grid, about 20 % of B, which a final
particlization could take; the grid edges, 6 %) and iSS sampling less B
than the outflow it gets (8-11 % in these single samples; the mean of 20
samples: 7 % with the drain, 3 % without; iSS's hadron gas against NEOS-4D). `drain_particlized 0` (the default of
MUSIC) changes nothing: the dynamic event and the staged Milne one are
byte-identical to before.

## Not done

- Upstream confirmation of the guards (MUSIC-fluid/MUSIC branch `XSCAPE`,
  `cee9460`, where the same lines sit at the same line numbers).
- Transporting W^{zz} directly in Cartesian mode (see above).
- Cartesian on Metal.
- Particlizing the fluid left on the grid at the end of a drained run (it
  is then matter that was never particlized), and the B deficit of iSS's
  sampling against the outflow it gets.
- The Cartesian diffusion terms on the GPU are checked only for Milne
  bit-identity; the blob has no baryon density. The SMASH runs of Track B
  (B3/B4, NEOS-4D with B, Q, S) are the test with charges.
