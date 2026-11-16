# B, Q, S charges and NEOS-4D on the GPU (CUDA)

Status: November 2026, branch `bqs_gpu` (based on `XSCAPE` at `1c99bee`).
Track A of the "Cartesian MUSIC on GPU" plan: the staged SMASH → MUSIC →
SMASH setup of X-SCAPE (`SMASHInitialCondition`) on the GPU with net baryon
number, electric charge and strangeness, the NEOS-4D equation of state (EOS 20)
and baryon diffusion.

Before this branch every one of these sent the whole step back to the CPU
(`gpu_features_supported`, `gpu_charges_ok`, PORT_GPU.md §4.1, §4.2). Now,
with a CUDA build:

| Feature | GPU |
|---|---|
| EOS 20 (NEOS-4D) | device lookup of P, T, μ_B, μ_Q, μ_S, c_s and ∂P/∂(e, ρ_B, ρ_Q, ρ_S) |
| `Include_Rhob`, `Include_QS` | 5 or 7 conserved components in the KT step; ρ_Q, ρ_S in the device state |
| B, Q, S source terms | pre-filled on the host into channels 4, 5, 6 of the source buffer |
| shear and bulk viscosity | T, s, c_s², μ_B of the cell from the 4D lookup |
| μ_B-dependent η/s (modes 7, 10) and ζ/s (mode 10) | yes |
| `turn_on_baryon_diffusion` | yes (not with `Include_vorticity_terms`, not for `Initial_profile 1`) |
| freeze-out | the fast gather carries ρ_Q, ρ_S (23 fields); the surface carries q^μ |
| Metal | unchanged: these cases stay on the CPU |

Every other EOS takes the GPU path exactly as before: the 1D-EOS kernels are
textually unchanged, and an EOS 91 run (`tests`-style 201×201×9 viscous Gubser
profile with bulk, freeze-out and evolution output) gives byte-identical output
files to `1c99bee`.

## Design

**Device EOS** (`src/gpu/eos4d_device.cuh`). A port of
`EOS_4D::get_tilde_variables` and `EOS_4D::FourDLInterp`: the closed-form map
(e, ρ_i) → (T̃, μ̃_B, μ̃_Q, μ̃_S), then one 16-corner interpolation. The six
tables are uploaded once, interleaved per node (`GPUGrid::eos4d_tab`, 421 MB),
so one corner fetch returns all six values. Float throughout, with the CPU's
clamping. The derivatives are formed from corner-pair differences, which keeps
them accurate in float; `FourDLInterp` subtracts two face sums instead and
loses most digits (see the self-test below).

**Ideal step.** Separate kernels `gpu_make_delta_qi_eos4d` and
`gpu_finalize_ideal_eos4d`. The reconstruction solves for v with
ρ_i = J_i √(1 − v²) inside the residual and an analytic Newton derivative
(the form `reconst.cpp` keeps commented out), bracketed in [0, 1]; a revert
keeps the previous cell's charges (`Reconst::revert_grid`), a regulated cell has
none. The signal speed takes c_s² from the table, as `Advance::MaxSpeed`.
Inlining the 4D code into the existing 1D kernels broke their 1D branch on the
GB10 (CUDA 13: non-finite fluxes with no error); separate kernels avoid that
and keep the 1D kernels byte-identical.

**Viscous step.** `gpu_first_rk_step_w_full_eos4d`, a copy of the 1D kernel
whose source terms take the cell's thermodynamics (P, T, s, c_s², μ_B) from one
4D lookup, with the μ_B dependence of η/s and ζ/s. Unlike the 1D kernel it
evolves Π and q^μ when shear is off (the 1D kernel zeroes W and Π then; with
shear off and bulk on, the 1D GPU path gives Π = 0 where the CPU evolves it).

**Diffusion.** `gpu_muBT_eos4d` (μ_B/T of the current and previous step),
`gpu_make_dmuBT` (∂^μ(μ_B/T) and u·∂(μ_B/T), as `U_derivative`),
`gpu_make_uwrhs_q` (KT flux of u^a q^i), and the q update in the viscous kernel
(`Diss::Make_uqSource`, transversality, `QuestRevert_qmu`). ∂_a q^{aμ} enters
through the existing `gpu_make_w_source`, which already loops over component 4.

**Host.** `Advance::eos4d_on_gpu()` gates it (CUDA, EOS 20,
`MUSIC_GPU_EOS4D` not 0, tables uploaded). `GPUGrid::with_qs` adds ρ_Q, ρ_S to
the uploads, copies, checkpoints and the freeze-out gather.

## Switches

| Environment | Effect |
|---|---|
| `MUSIC_GPU_EOS4D=0` | EOS 20 stays on the CPU (as before this branch) |
| `MUSIC_EOS4D_SELFTEST=N` | at GPU start-up, compare the device lookup with the CPU EOS at N points |
| `MUSIC_DUMP_STATE=<file>` | write every cell's e, ρ_B, ρ_Q, ρ_S, u, W, Π (float64) at the end of the run |
| `MUSIC_DUMP_STATE_STEP=N` | with the above: after time step N of a time-stepped run, to `<file>.stepN` |

## Validation (GB10)

**Lookup**, `MUSIC_EOS4D_SELFTEST=1000000` (points over the whole table and 10 %
beyond each edge): P, T, s agree with the CPU to 1–9·10⁻⁶ at the maximum,
c_s² to 2·10⁻⁴, μ_B, μ_Q, μ_S to 3·10⁻⁵ at the 99.9th percentile (the maxima,
10⁻³–10⁻², are at values crossing zero). Against the multilinear interpolant in
double, the device ∂P/∂ρ_i agree to 10⁻⁶ (median) and 10⁻³ (99.9 %); the CPU's
own `FourDLInterp` derivatives are off by O(1) there (float cancellation).

**Ideal B, Q, S**: Gubser flow with the ρ_B of `Gubser_solution_ideal.py`,
ρ_Q = 0.4 ρ_B, ρ_S = 0.1 ρ_B, EOS 20, 50 steps: every field within 7·10⁻⁵ of its
maximum, totals within 2·10⁻⁵, the ratios preserved.

**Viscous**: the viscous Gubser profile (201×201×9) with EOS 20, η/s mode 3 and
μ_B mode 10, bulk mode 10, 20 steps: e 1.6·10⁻⁵, u 9·10⁻⁵, W 8·10⁻⁴, Π 10⁻³
(relative to the maximum).

**Diffusion**: the ideal Gubser profile with ρ_B, Q, S and baryon diffusion
(C_B = 0.4): after one step every field within 4·10⁻⁵; after 50 steps the
totals agree (q^τ to 4·10⁻⁵) and the cells agree except within 0.2 fm of the
centre, a symmetric extremum of μ_B/T at e = 0.08 fm⁻⁴ (0.016 GeV/fm³) where
the minmod limiter turns float noise into a small asymmetric wiggle.

## CPU defects found on the way (fixed here)

1. `EOS::get_pressure` and `get_dpdrhob` forwarded only (e, ρ_B): with EOS 20
   the pressure was taken at ρ_Q = ρ_S = 0 while T, μ and c_s² were not.
2. `EOS_base::getThermalVariables` formed the entropy with μ_S ρ_Q + μ_Q ρ_S
   (slots 8 and 10 swapped); `get_entropy` had it right. It enters η = (η/s) s.
3. `TransportCoeffs::get_muB_dependence_shear_profile` (η/s mode 10) raised a
   negative μ_B to a fractional power: NaN, clipped to η/s = 0, and the WW
   term's τ_π/η became infinite. It now uses |μ_B|, like mode 7.
4. `Evolve::EvolveOneTimeStep` took the source end time from the main source
   only; X-SCAPE hands its sources (the hadronic liquefier of the SMASH initial
   state) to the jet slot, so the "energy density increased by more than a
   factor of 2" abort fired on the empty grid when the hadrons arrived: the
   SMASH-IC example did not run at all.

Items 1 to 3 change CPU results for EOS 20 and for η/s mode 10; nothing else.
