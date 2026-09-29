# GPU grid freeze in dilute regions (vacuum cells with huge u⁰)

**Status:** fixed on branch `nonfinite_guard` (commit `f73ffab`).

**Affects:** the MUSIC4GPU CUDA path. The Metal path has the same code; it is validated on an
M3 Max (see [Metal (M3 Max)](#metal-m3-max)), where the failing event does not fail.

**Found in:** an X-SCAPE production, 0–10% Au+Au 200 GeV, 3D MC-Glauber strings
(`InitialProfile 131`), viscous, hotQCD EoS.

## Summary

Once in 40 background runs, a MUSIC run on the GPU stopped evolving part-way through, and
nothing reported it. The grid kept stepping in τ, but the fields no longer changed. MUSIC ran
to its maximum time (`⚠ Maximum allowed time reached.`) and returned a stalled evolution as if
it were a normal one. The same event on the CPU path (`MUSIC_FORCE_CPU=1`) froze out normally.

The cause is single precision in dilute regions:

1. vacuum cells with huge Lorentz factors carry real energy-momentum on the GPU;
2. one of them drags the dilute edge of the fluid to u⁰ ≈ 190 in one step;
3. the viscous stress overflows there;
4. the resulting NaN spreads until every cell is reverted every step.

The fix puts vacuum-level cells at rest. It also guards the stored viscous state against
non-finite values and counts how often that guard acts.

## Symptoms

- **The run ends at the maximum time.** The log shows `⚠ Maximum allowed time reached.`
  instead of `Finished at tau = …`.
- **The stored evolution freezes part-way.**
  - From one frame on (here τ ≈ 5.1 fm/c), e and u stop changing: the largest change
    between frames is at rounding level, 2×10⁻⁷ GeV/fm³.
  - The hottest cell stays at a fixed value (here 2.805 GeV/fm³), so the maximum energy
    density never falls below e_fo.
  - The run therefore fills the whole output window (here 300 frames, to τ = 30.5 fm/c).
- **The stored fields show nothing wrong.** e and u stay finite, because every failing cell is
  reverted to its previous e and u, and W^{μν} and Π are not part of the output.
- **Downstream, energy and momentum are wrong.** A grid that stopped changing while τ grows
  gains energy through the τ in the volume element. In the X-SCAPE pair production this showed
  up as wake energies of −5×10⁴ to −8×10⁴ GeV: jet leg − background, where the background had
  "grown" to about 10⁵ GeV.

**How often:** once in 40 background (MUSIC_1) runs across two productions of 20 jobs each.
None of the 600 jet-leg runs on the same initial conditions were affected. The failure depends
sensitively on the exact state: the legs' rounding-level differences grow chaotically and are
carried across the grid by the numerical stencil, so a jet leg starting from the same initial
condition reaches the dilute edge in a slightly different state.

## Reproduction

With js-contrib's `run_prod_jet.py`, one background and three jets, about 2 minutes on a GB10:

```sh
cd js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
python run_prod_jet.py --events 3 --seed 711178985 --pthat-bins 10-20,20-30,30-40 \
    --jets-per-bin 1 --parton-ymax 0.6 --parton-y-mode leading --bg-layout shared \
    --outdir /tmp/repro --out repro.h5 --seed-registry none
grep -E "Maximum allowed time|Finished at tau" /tmp/repro/*.log
```

Before the fix, MUSIC_1 logs `Maximum allowed time reached.` and all three jet legs finish
normally. The background is bit-identical on every rerun.

## Root cause

### 1. Vacuum cells with huge Lorentz factors carry energy on the GPU

Both paths produce vacuum cells moving at ultra-relativistic speeds at the dilute edge of the
medium. What differs is how much energy-momentum they carry, which grows as T^{ττ} ∝ (e + P)·u⁰².
With `MUSIC_DEBUG_NONFINITE=10` on the same event:

| path | vacuum floor of e | cells with u⁰ > 100 | largest u⁰ | T^{ττ} of such a cell |
|---|---|---|---|---|
| CPU (double) | ~10⁻¹⁴ GeV/fm³ | ~1000–4000 | ~6000 | ~10⁻⁷ GeV/fm³: harmless |
| GPU (float) | ~10⁻⁷ GeV/fm³ | ~50–70 | 2048 | **~1 GeV/fm³** |

The seed cell of the failure has e ≈ 0.005 GeV/fm³ and T^{ττ} ≈ 0.08 GeV/fm³. A single GPU
vacuum cell next to it therefore carries more than ten times its energy-momentum.

### 2. The fluid's dilute edge is dragged along

With `MUSIC_DEBUG_CELL=49,67,54,175,197`, which is MUSIC grid cell (x, y, η_s) ≈ (−0.3, 5.1, 4.8):

| step (τ) | cell | e [GeV/fm³] | u⁰ | u¹ | u² | u³ |
|---|---|---|---|---|---|---|
| 189 (4.18) | ix 48, vacuum | 1.9×10⁻⁷ | **2048** | −983 | 282 | −1774 |
| 189 | ix 49, seed | 0.0055 | 3.86 | −1.87 | 0.53 | −3.19 |
| 190 (4.20) | ix 49, seed | 0.0057 | **190.7** | −91.6 | 26.2 | −165.2 |
| 191 (4.22) | ix 50 | 0.011 | 13.95 | −6.71 | 1.92 | −12.04 |

The vacuum neighbour at ix 48 does not change from step to step: it is reverted every step
and keeps u⁰ = 2048. At step 190 the seed cell's four-velocity becomes 0.093 times the
neighbour's, i.e. exactly parallel to it. The next cell follows one step later. The kernel's
"runaway reconstruction" check rejects only jumps of more than 100×, as the CPU does
(`check_u0_var`), so the 49× jump passes.

### 3. The viscous stress overflows, and the regulator makes it NaN

The velocity gradients next to the seed cell drive W^{μν} and Π to infinity. The QuestRevert
regulator in `gpu_first_rk_step_w_full` cannot remove that:

- the shear tensor is scaled by `rho_shear_max / rho_shear`, which is 0 for an infinite
  size, and ∞ × 0 = NaN;
- for the bulk pressure, `rho_bulk > rho_bulk_max` is false for NaN, so a NaN Π is passed
  through unchanged.

`MUSIC_DEBUG_NONFINITE=5` counts the non-finite values:

| step (τ) | non-finite W^{μν} | non-finite Π |
|---|---|---|
| 190 (4.20) | 0 | 0 |
| 195 (4.30) | 1,544 | 999 |
| 200 (4.40) | 19,712 | 16,679 |
| 210 (4.60) | 138,848 | 160,239 |
| 235 (5.10) | 280,594 | 582,364 |
| 250 (5.40) | 280,594 | 600,000 (all cells) |

### 4. The whole grid is reverted, every step

A non-finite value reaches the neighbouring cells through the flux stencil, about four cells
per step. `gpu_reconst` then reverts each affected cell to its previous e and u: non-finite
T^{ττ}, T^{ττ} < |M|, or no velocity root. It does not revert W^{μν}, which stays non-finite.
So each affected cell stays frozen, and within about 40 steps the whole grid is. e and u stay
finite throughout, which is why the output looks normal.

### 5. Nothing stops it

In beast mode (`beastMode 1`, production) MUSIC runs no per-step state checks, so it steps to
its maximum time. Even `beastMode 0` would not help: its checks look for a *growing* maximum
energy density, and here it just stops changing.

**Ruled out as causes:**

- the GPU frame-packing kernel: `MUSIC_GPU_NO_PACK=1` freezes identically;
- the freeze-out surface finder: a run without the surface freezes identically;
- X-SCAPE's liquefier: MUSIC_1 has no jet source, and X-SCAPE `contrib` gives a bit-identical
  failure;
- transient GPU errors: the failure is bit-reproducible, and later runs in the same process
  are fine.

## The fix

### Vacuum at rest (`gpu_reconst`, CUDA and Metal)

After the reconstruction, a cell with

    e < GPU_VACUUM_E = 1e-5 1/fm^4   (≈ 2×10⁻⁶ GeV/fm³, 10⁵ below freeze-out)
    and u⁰ > GPU_VACUUM_U0_MAX = 10

is put at rest: u = (1, 0, 0, 0), with its e kept. This removes the source: only the spurious
kinetic energy of a vacuum cell is dropped. Real fluid is never that dilute. The seed cell
above (e ≈ 0.028 1/fm⁴) is not touched; its vacuum neighbour (e ≈ 9.5×10⁻⁷ 1/fm⁴, u⁰ = 2048)
is.

`gpu_reconst` wraps the original function (renamed `gpu_reconst_raw`), so every return path
is covered. That includes the reverts and the half-step states at the cell faces used for the
fluxes.

### Non-finite guard (`gpu_first_rk_step_w_full`; Metal: both W kernels)

Before the viscous state is stored, a non-finite W^{μν} (all 14 components) or Π is set to 0.
Only non-finite values trigger it, so a run that never overflows is bit-identical to before.

### Counter and warning (CUDA)

- **Counting:** the kernel counts such cell updates in a `__device__` global with `atomicAdd`,
  only when the guard acts. That costs no extra memory traffic.
- **Reporting:** `Evolve` reads the count every 10 steps in both evolution loops (`EvolveIt`
  and `EvolveOneTimeStep`). It warns at the first non-zero count and reports the total at the
  end of the run:

      ⚠ GPU: N cell update(s) with a non-finite W^{mu nu} or Pi, set to 0 (up to step s, tau = …)

- **Aborting:** `MUSIC_ABORT_ON_NONFINITE=1` stops the run instead.
- **Reset:** the count is reset when the GPU takes over the state at a run's first step, so it
  does not carry over into the next run of the same process.
- **Metal:** the same count, with an `atomic_uint` in a 4-byte shared buffer bound as
  buffer 24 of `gpu_first_rk_step_w_full` (Metal has no device globals).
  `MetalPipelines::read_and_reset_nonfinite()` waits for the last committed command buffer,
  then reads and zeroes it. Counts from a still-open batch arrive at the next read.

The guard alone is not enough. With it but without the vacuum reset, the NaN no longer freezes
the grid, but the fluid next to the u⁰ = 2048 vacuum cell blows up (e → 10¹⁹ GeV/fm³). It then
reaches the grid edge, and MUSIC stops the run there (`Freeze-out cell at the boundary!`).

### Debug switches (off unless set)

| variable | effect |
|---|---|
| `MUSIC_DEBUG_NONFINITE=N` | every N steps: counts of non-finite e, u, W^{μν}, Π, ρ_B, with the first location; the largest \|W^{μν}\|/e; the number of cells with u⁰ > 100 and the largest u⁰, with its e and location |
| `MUSIC_DEBUG_CELL=ix,iy,ieta,from,to` | every step in [from, to]: e [GeV/fm³], u^μ, Π and all 14 W^{μν} components of that cell and its two x neighbours. Indices are MUSIC's grid: cell centre −size/2 + i·d |

- **Cost when unset:** one integer comparison per step.
- **Cost when set:** each check copies the whole GPU state to the host, a debugging cost.
- **Where:** both evolution loops.

## Validation

**The failing event** (the reproduction above):

| | before | guard only | guard + vacuum reset | CPU path |
|---|---|---|---|---|
| background ends at | 30.5 fm/c (max time) | 6.4 fm/c (grid edge) | **10.86 fm/c** | 10.86 fm/c |
| non-finite W^{μν}/Π | 600,000 cells | zeroed (643 by step 200) | 0 | 0 |
| cells with u⁰ > 100 | 50–70 | 50–70 | 0 | 1000–4000 (harmless) |
| max e at the end | 2.805 (stuck) | 2.5×10¹⁹ | 0.228 (≈ e_fo) | ≈ e_fo |

**A normal event** (job 0003 of the same production, 15 events, same seed, compared with the
build without these changes):

- **Guard only:** bit-identical (background, all 15 jet legs, all droplets); no warnings.
- **Guard + vacuum reset:** the results change slightly, since every run has vacuum cells with
  u⁰ > 10:

  | quantity | change |
  |---|---|
  | background freeze-out time | identical (10.4 fm/c) |
  | largest \|Δe\| anywhere | 0.007 GeV/fm³ |
  | largest \|Δe\|/e in cells above freeze-out | 1% |
  | background energy at τ = 7.5 fm/c | −0.03% (33,487 → 33,478 GeV) |
  | jet-leg freeze-out times (15 events) | identical |
  | wake energy per event at τ = 7.5 fm/c | ±0.6 GeV of ~40 GeV |
  | wake energy / deposited energy, median (16–84%) | 1.02 (0.95–1.06) → 1.01 (0.95–1.05) |

  Showers and droplets differ event by event, because the energy loss (MATTER, LBT) samples the
  slightly different background, so their random sequence goes a different way. Statistically
  they are the same.

**Cost:** the vacuum reset is one comparison per reconstruction, and the counter only adds work
when the guard acts. Neither is measurable next to a ~30 ms step.

### Metal (M3 Max)

Tested 2026-09-28 on an Apple M3 Max: X-SCAPE `contrib` 4a7a158f, this branch at 6d171df,
js-contrib `main` 53ab4ef, built with `USE_METAL=ON`. The reproduction above was run with the
build's kernels, with variant kernels selected by `MUSIC_METALLIB` (the pre-fix
`music_kernels.metal` of 6b238c4, and this one without the vacuum reset: all three pieces of the
fix are in the shader, so the libraries stay the same), and on the CPU path.

**The failing event does not fail on Metal.** Metal's rounding differs from CUDA's, and its
vacuum cells are much slower, so the chain in [Root cause](#root-cause) does not start:

| | before | guard only | guard + vacuum reset | CPU path |
|---|---|---|---|---|
| background ends at | 10.56 fm/c | 10.56 fm/c | 10.56 fm/c | 10.56 fm/c |
| jet legs end at | 12.46, 10.86, 10.56 | same | same | 12.46, 10.86, 11.26 |
| non-finite W^{μν}/Π | 0 | 0 | 0 | 0 |
| cells with u⁰ > 100 (most at once) | 98 | 98 | 1 | ~3300 (harmless) |
| largest u⁰ | 337 | 337 | 132 | ~6300 |

The background ends at the same τ on Metal and on the CPU; the 10.86 fm/c above is the GB10
value. The one remaining cell with u⁰ > 100 (step 40) has e = 2.1×10⁻⁶ GeV/fm³, just above the
reset threshold, and is gone ten steps later.

**The fix changes Metal results at the same level as on CUDA** (the reproduction, 1 background
and 3 jets, guard + vacuum reset against before):

| quantity | Metal | CUDA (above) |
|---|---|---|
| guard only | bit-identical hydro output | bit-identical |
| freeze-out times (background, 3 jet legs) | identical | identical |
| largest \|Δe\| anywhere | 0.006 GeV/fm³ | 0.007 GeV/fm³ |
| largest \|Δe\|/e in cells above freeze-out | 0.6% | 1% |
| background energy at τ = 7.5 fm/c | −0.035% (30,643 → 30,633 GeV) | −0.03% |
| wake energy per event at τ = 7.5 fm/c | +0.1 to +0.8 GeV of 15–26 GeV | ±0.6 GeV of ~40 GeV |

Against the CPU path, the fix moves the background energy at τ = 7.5 fm/c from +0.030% to
−0.004%, which is the spurious energy of the vacuum cells removed. The field-level agreement
with the CPU is unchanged (energy-weighted \|Δe\|/e = 1.8×10⁻³ with and without).

**The guard and the counter work on Metal.** Metal compiles with fast math by default, but
`isfinite` is kept (Apple metal 32023.864): a test kernel with the regulator's ∞ × 0 and a
division by 0 zeroes both and counts 2 of 2. A test shader that puts a NaN into W^{μν} and an
∞ into Π in two central cells for τ < 0.47 fm/c gives:

- the warning at the first check (step 10) and the total at the end of each run;
- 10 counted cell updates in each of three MUSIC runs in one process, so the count is reset
  between runs;
- freeze-out times identical to the run without the injection;
- with `MUSIC_ABORT_ON_NONFINITE=1`, a stop at step 10 (exit status 1).

**Cost and reproducibility:** 55.8 s with the fix and 55.6 s without (3 events, 2 runs each).
Reruns are bit-identical in the hydro output, also with a different `OMP_NUM_THREADS` and with
the debug switches on.

## Impact on existing data

- **GPU runs made before this fix** can contain such events:
  - one or both legs are stalled and never freeze out;
  - they fill the output window;
  - their last frame is still hot.
- **How to find them:**
  - the job log contains `Maximum allowed time reached`;
  - js-contrib's `analysis/wake_observables.py` flags them (`jet_no_freezeout` /
    `bg_no_freezeout`: the hottest cell in the last frame is above 1.5 e_fo), and the notebook
    leaves them out.
- **Repair:** rerun the job with the fixed build. With the same seed, everything except the
  hydro legs is reproduced.
- **Other events:** they differ from a rerun with the fix only at the level shown under
  Validation.

## Open points

- **Metal:** validated on an M3 Max (see [Metal (M3 Max)](#metal-m3-max)). Since the failing
  event does not fail there, Metal has no case in which the fix is needed yet; the guard and
  the counter were exercised by injecting non-finite values.
- **Counter reset between runs:** exercised on Metal with injected non-finite values (see
  above). On CUDA, with the vacuum reset, no test case produces non-finite values any more.
- **Thresholds:** `GPU_VACUUM_E` and `GPU_VACUUM_U0_MAX` are compile-time constants, chosen so
  real fluid is never touched. They could become input parameters if other setups need
  different values.
- **Jump guard:** the "runaway reconstruction" check still rejects only jumps of more than 100×,
  as on the CPU. With the vacuum reset it no longer needs to catch this case.

## Files

| file | change |
|---|---|
| `src/gpu/music_kernels.cu` | `gpu_reconst` wraps `gpu_reconst_raw` with the vacuum reset; non-finite guard and `d_nonfinite_count` in `gpu_first_rk_step_w_full`; `gpu_nonfinite_count_read_and_reset()` |
| `src/gpu/music_kernels.cuh` | declaration of `gpu_nonfinite_count_read_and_reset()` |
| `src/gpu/music_kernels.metal` | vacuum reset; guard in `gpu_first_rk_step_w` and `gpu_first_rk_step_w_full`, the counter (`atomic_uint`, buffer 24) in the latter |
| `src/gpu/CUDAPipelines.{h,cu}` | `read_and_reset_nonfinite()` |
| `src/gpu/MetalPipelines.{h,mm}` | `read_and_reset_nonfinite()`; the 4-byte counter buffer, bound at buffer 24 |
| `src/advance.{h,cpp}` | `Advance::nonfinite_count_gpu()`; count reset when the GPU takes over the state |
| `src/evolve.{h,cpp}` | `check_nonfinite_gpu()` and `debug_hooks()` in both evolution loops; the debug switches |
| `README_CUDA.md` | short section on the mechanism, the fix and the switches |
