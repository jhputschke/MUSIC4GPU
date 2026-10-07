# MUSIC on the CPU vs. MUSIC4GPU: what the GPU path changes

> Copy of js-contrib [`docs/MUSIC_CPU_vs_GPU.md`](https://github.com/jhputschke/js-contrib/blob/main/docs/MUSIC_CPU_vs_GPU.md)
> (js-contrib PR #47, 2026-10-07). The production it uses (X-SCAPE + PyJetscape,
> `prod_AuAu_0_10_jet`) and the analysis scripts are in js-contrib; the links below point
> there. The test ran MUSIC4GPU `49439c0` (this repository's `XSCAPE` branch) on the GB10
> (CUDA).

**Question.** MUSIC4GPU evolves in single precision and puts vacuum cells moving with
u⁰ > 10 at rest (PR #13, [`VacReset_BUG.md`](VacReset_BUG.md)). The CPU path of the same build
(`MUSIC_FORCE_CPU=1`) evolves in double precision and has no such reset. How large are the
differences, and do they matter for the physics a production delivers?

**Answer.** The two agree to about 0.2–0.3%, with identical freeze-out times. The residual
difference is systematic, though: on the GPU every event has about 0.3% less energy above
freeze-out and 0.4% more in the dilute fluid below it, a 0.19% smaller freeze-out volume, and
0.2% fewer hadrons. ⟨pT⟩, v₂, v₃, the momentum anisotropy and the jet wake agree within
their errors. Within one campaign that is negligible. Mixing GPU and CPU samples, or
comparing a GPU campaign with CPU results, brings in a 0.2% normalization offset that does
not average out.

*October 2026, GB10. 6 jet events on 3 backgrounds: see [Limits](#limits).*

## Method

1. **Same events on both paths.** `run_prod_jet.py` twice with identical options,
   `--native --events 6 --reuse 2 --seed 11 --write-particlize both`, the second time with
   `MUSIC_FORCE_CPU=1`. Both write MUSIC's own grid (0.3 fm × 0.3 fm × 0.2, a frame every
   0.1 fm/c) and both legs' freeze-out surfaces. The initial conditions, hard processes and
   string sources are the same; only MUSIC's arithmetic differs.
2. **Hydro** ([`music_cpu_gpu_hydro.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/music_cpu_gpu_hydro.py)):
   - backgrounds cell by cell;
   - energies of the ideal T^μν through the τ = const surface in the lab frame (the store has
     no viscous fields), split at e_fo = 0.2342 GeV/fm³ (T = 0.15 GeV, EOS 9) into the hot
     region and the dilute fluid;
   - the momentum anisotropy ε_p = Σ(T^xx − T^yy)/Σ(T^xx + T^yy) at |η_s| < 0.5, the hydro
     precursor of v₂;
   - vacuum cells counted, not summed (below).
3. **Surfaces and hadrons** ([`music_cpu_gpu_hadrons.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/music_cpu_gpu_hadrons.py)):
   - the surface summaries of [`hydro_hist_vs_surface/analyze.py`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/hydro_hist_vs_surface/analyze.py);
   - iSS on both paths' surfaces with the same seeds: the CPU particlize file gets the GPU
     file's `file_uuid` (`--match-seeds`), then `hadronize.py --tags bulk_jet,bulk_bg
     --oversample 200 --correlated` on both. With correlated sampling iSS addresses its
     random numbers by the cell, so the two give the same hadrons wherever their surfaces
     agree, and the difference is measured per oversample with little noise.

**Energies count real fluid only** (e ≥ 10⁻⁴ GeV/fm³). The stored velocities are lab-frame
float32 values, v = u/u^t. The CPU's vacuum cells (e ≈ 10⁻¹⁴ GeV/fm³) move with u^τ up to
~6000, so their velocities round to |v| = 1 and their u^t cannot be recovered from the file.
At e ≥ 10⁻⁴ GeV/fm³ at most 2 cells of a frame are affected. (A first version that computed
γ for every cell gave the CPU thousands of GeV of spurious "vacuum energy".)

**Events.** 0–10% Au+Au 200 GeV, PythiaGun pT̂ 50–70 GeV, Matter + LBT and the
CausalLiquefier (`AuAu_MCGlauber_MUSIC_0_10_jet.xml`). X-SCAPE `cbc72639`, MUSIC4GPU
`49439c0`, js-contrib `337b9c2`. GPU job 3.2 min, CPU job 23 min (20 cores).

## Results

### Hydro: backgrounds cell by cell

All three backgrounds behave the same way. Medians over the frames with τ ≥ 1 fm/c:

| quantity | GPU vs. CPU |
|---|---|
| freeze-out time, all 9 MUSIC runs (3 backgrounds, 6 jet legs) | identical (10.46 … 12.86 fm/c) |
| \|Δe\|/e above freeze-out, energy-weighted | 2.1×10⁻³ (single cells up to 3.6×10⁻²) |
| \|ΔT\|/T above freeze-out | 2.9×10⁻⁴ |
| \|Δv\| above freeze-out | 1.6×10⁻⁴ |
| fluid energy (e ≥ 10⁻⁴ GeV/fm³) | +0.02% to +0.08% |
| energy above freeze-out | **−0.30% to −0.34%** |
| energy of the dilute fluid (10⁻⁴ ≤ e < e_fo) | **+0.39% to +0.45%** |
| momentum anisotropy ε_p | equal to ≤ 6×10⁻⁵ absolute (ε_p = −0.018 … +0.045) |
| vacuum cells with u^τ > 10 (most in one frame) | GPU 12–13, CPU 17,000–19,000 |

Background 0 frame by frame ((G − C)/C):

| τ [fm/c] | fluid energy [GeV] | fluid | hot | dilute | fast vacuum cells GPU / CPU | ε_p GPU / CPU | \|Δe\|/e |
|---|---|---|---|---|---|---|---|
| 1 | 35,190 | +4.3×10⁻⁴ | +1.9×10⁻⁴ | +9.7×10⁻³ | 1 / 294 | 0.0015 / 0.0015 | 1.4×10⁻³ |
| 2 | 34,862 | +6.7×10⁻⁴ | +0.9×10⁻⁴ | +1.1×10⁻² | 3 / 3,822 | 0.0144 / 0.0145 | 2.2×10⁻³ |
| 4 | 34,785 | +3.4×10⁻⁴ | −1.8×10⁻³ | +7.7×10⁻³ | 1 / 7,822 | 0.0362 / 0.0363 | 2.2×10⁻³ |
| 6 | 34,775 | −0.2×10⁻⁴ | −3.4×10⁻³ | +3.5×10⁻³ | 3 / 10,341 | 0.0453 / 0.0453 | 2.1×10⁻³ |
| 8 | 34,731 | +0.1×10⁻⁴ | −6.3×10⁻³ | +1.9×10⁻³ | 6 / 14,380 | 0.0444 / 0.0444 | 1.8×10⁻³ |
| 10 | 34,637 | +2.3×10⁻⁴ | −2.5×10⁻² | +0.4×10⁻³ | 6 / 16,888 | 0.0432 / 0.0433 | 1.1×10⁻³ |

At τ = 6 fm/c that is about 60 GeV, 0.17% of the total, moved from the hot region into the
dilute fluid on the GPU. The late relative values of the hot energy are large only because
little hot matter is left.

- **Same flow, slightly different energy distribution.** The total fluid energy agrees to
  < 0.1%, ε_p to ≤ 10⁻⁴. On the GPU a little energy moves from the core into the dilute edge.
- **The vacuum reset works.** The CPU keeps thousands of vacuum cells moving with
  u^τ > 10; on the GPU at most 13 remain (with e just above the reset threshold). On the CPU
  these cells carry no measurable energy (vacuum floor ~10⁻¹⁴ GeV/fm³); on the GPU, before
  the fix, they did ([`VacReset_BUG.md`](VacReset_BUG.md)).

### Freeze-out surfaces

GPU/CPU − 1, the three backgrounds:

| quantity | bg 0 | bg 2 | bg 4 |
|---|---|---|---|
| cells | −1.3×10⁻³ | −1.4×10⁻³ | −1.4×10⁻³ |
| volume V = Σ dσ·u | **−1.9×10⁻³** | **−1.9×10⁻³** | **−2.0×10⁻³** |
| negative part of V | −4×10⁻² | −3×10⁻² | −3×10⁻² |
| ⟨u_T⟩ at \|η_s\| < 1 | −1.1×10⁻³ | −1.1×10⁻³ | −1.1×10⁻³ |
| ⟨√π:π⟩ / ⟨Π⟩ at \|η_s\| < 1 | −2.2 / −0.4×10⁻³ | −1.8 / −0.4×10⁻³ | −2.0 / −0.3×10⁻³ |
| shape of dV/dτ (Σ\|G − C\|/Σ\|C\|) | 3.0×10⁻³ | 3.1×10⁻³ | 3.2×10⁻³ |

The jet legs with the same showers on both paths show the same values. The negative part of
V is 0.01–0.03% of V.

### Hadrons

iSS with the same seeds, 200 oversamples, GPU − CPU paired per oversample. The backgrounds,
(G − C)/C ± the paired error:

| observable | bg 0 | bg 2 | bg 4 | independent sampling error |
|---|---|---|---|---|
| charged hadrons, \|η\| < 5 | **−2.4 ± 0.2** ×10⁻³ | **−2.0 ± 0.2** ×10⁻³ | **−2.0 ± 0.2** ×10⁻³ | 1×10⁻³ |
| dN_ch/dη, \|η\| < 0.5 | −1.8 ± 1.0 ×10⁻³ | −1.4 ± 1.0 ×10⁻³ | −2.6 ± 1.0 ×10⁻³ | 3×10⁻³ |
| dN/dy π⁺, \|y\| < 0.5 | −3.4 ± 1 ×10⁻³ | −0.9 ± 1 ×10⁻³ | −3.4 ± 1 ×10⁻³ | 4×10⁻³ |
| dN/dy K⁺, p | within ±9×10⁻³ (± 3–5×10⁻³) | | | 9×10⁻³, 1×10⁻² |
| ⟨pT⟩ π⁺, K⁺, p | within ±3×10⁻³ (± 1–3×10⁻³) | | | 3–8×10⁻³ |
| total hadron energy | +0.5 ± 0.3 ×10⁻³ | +0.9 ± 0.3 ×10⁻³ | +1.8 ± 0.3 ×10⁻³ | 2×10⁻³ |
| v₂ (CPU value), G − C | 0.029: −2×10⁻⁴ | 0.010: −9×10⁻⁴ | 0.011: +4×10⁻⁴ | 2×10⁻³ |
| v₃ (CPU value), G − C | 0.018: −5×10⁻⁴ | 0.008: −2×10⁻⁴ | 0.006: −3×10⁻⁴ | 2×10⁻³ |

v₂ and v₃ are charged, |η| < 1, 0.2 < pT < 3 GeV, from all oversamples of the event; their
error from the two halves of the oversamples is rough (a few ×10⁻⁴).

- **Multiplicity:** 0.2% fewer hadrons on the GPU, significant at 10σ over the full
  acceptance. It matches the 0.19% smaller freeze-out volume: at fixed T_fo the thermal
  yield scales with V.
- **⟨pT⟩, v₂, v₃:** no significant shift. The v_n differences (≤ 10⁻³ absolute) are below the
  sampling noise of 200 oversamples, as ε_p's equality on the hydro side suggests.
- **Total hadron energy:** +0.05 to +0.2%, the opposite sign. It is dominated by forward
  rapidities; possibly the GPU's extra dilute energy freezing out there, not followed up.

### Jets

- **Showers.** In 3 of the 6 events Matter and LBT produced the same showers on both paths
  (droplets within 2×10⁻³). In the other 3 they diverged: the background temperature differs
  by ~3×10⁻⁴, which is enough to send the energy loss's random sequence another way. That
  decorrelates single events; it is not a bias.
- **Same-shower events, wake Δe = e_jet − e_bg cell by cell:**

  | event (droplets) | τ [fm/c] | wake GPU / CPU [GeV] | \|Δe_G − Δe_C\| / \|Δe_C\| (L2) | peak Δe GPU / CPU [GeV/fm³] |
  |---|---|---|---|---|
  | 0 (35.0 GeV) | 4 / 6 / 10 | 11.47 / 11.48, 30.88 / 30.95, 35.79 / 35.56 | 0.001–0.002 | 1.317 / 1.318 at τ = 4 |
  | 1 (36.8 GeV) | 4 / 6 / 10 | 15.75 / 15.86, 27.61 / 27.95, 38.21 / 38.09 | 0.001–0.002 | 1.014 / 1.015 at τ = 4 |
  | 2 (6.0 GeV) | 4 / 8 / 10 | 4.44 / 4.39, 6.06 / 5.34, 6.11 / 5.45 | 0.003–0.011 | 0.485 / 0.485 at τ = 4 |

  The fields agree to 0.1–1%. The integrated wakes agree to about 1% (0.1–0.3 GeV) for the
  35 GeV jets; the 6 GeV wake differs by up to 0.7 GeV late, which is the size of the
  energy moving between the hot and the dilute region of a 34,000 GeV background.
- **All events, wake per deposited energy:** GPU 1.033 ± 0.012, CPU 1.017 ± 0.022 (6 events
  each, partly different showers).
- **The medium the jet sees** differs by ~3×10⁻⁴ in T, i.e. ~10⁻³ in q̂ ∝ T³.

## Physics implications

- **Within one campaign: negligible.** A 0.2% change in multiplicity and freeze-out volume
  is far below the model and tuning uncertainties, and below the accuracy expected of an
  FNO trained on the data. Flow, ⟨pT⟩, the medium temperature and the wake are unchanged
  within errors.
- **Across paths: a normalization offset.** The shift has the same sign in every event, so it
  does not average out. With ~10⁴ events the statistical error of dN_ch/dη reaches ~0.1%,
  where the offset shows. Keep a campaign, and an FNO training set, on one path, and name
  the path when comparing with CPU MUSIC or published MUSIC results.
- **Single events differ in their jets**, not in their bulk: the same seed can give a
  different shower on the other path. Compare paths statistically, or through the events
  whose droplets match (`music_cpu_gpu_hydro.py` lists them).

**Cause.** This test cannot separate single precision from the vacuum reset or from the GPU
path's EoS table. MUSIC4GPU's own validation measured the reset alone at ~0.03% of the
background energy, with the field-level agreement with the CPU unchanged with and without it
(energy-weighted |Δe|/e = 1.8×10⁻³ on Metal, [`VacReset_BUG.md`](VacReset_BUG.md)). Most of the 0.2–0.3% is
therefore probably single precision, presumably extra numerical diffusion from the core
into the dilute edge. That is an inference, not a measurement; a GPU build without the reset,
run on these events, would settle it.

## Limits

- 6 jet events on 3 backgrounds, one centrality (0–10%), one pT̂ window.
- The energies are those of the ideal T^μν: the store has no viscous fields.
- Hadrons from iSS only (no afterburner, no jet fragmentation); v_n from single events with
  200 oversamples each.
- The cause of the systematic offset is not isolated (above).

## Reproduce

In an X-SCAPE checkout with js-contrib (`external_packages/js-contrib`, its
[`prod_AuAu_0_10_jet`](https://github.com/jhputschke/js-contrib/blob/main/contribs/PyJetscape/example/prod_AuAu_0_10_jet/README.md) production) and a MUSIC4GPU build:

```bash
conda activate js_fno
cd external_packages/js-contrib/contribs/PyJetscape/example/prod_AuAu_0_10_jet
OPTS="--native --events 6 --reuse 2 --seed 11 --write-particlize both --seed-registry none"
python run_prod_jet.py $OPTS --outdir OUT/gpu                      # 3.2 min on the GB10
MUSIC_FORCE_CPU=1 python run_prod_jet.py $OPTS --outdir OUT/cpu    # 23 min
python music_cpu_gpu_hydro.py OUT/gpu/AuAu_0_10_jet_seed0011.h5 OUT/cpu/AuAu_0_10_jet_seed0011.h5

python music_cpu_gpu_hadrons.py --match-seeds OUT/gpu OUT/cpu      # same iSS seeds
for d in gpu cpu; do
  python hadronize.py OUT/$d/AuAu_0_10_jet_seed0011_particlize.h5 \
      --tags bulk_jet,bulk_bg --oversample 200 --correlated
done
python music_cpu_gpu_hadrons.py OUT/gpu OUT/cpu
```

The two jobs write ~10 GB (both legs on MUSIC's grid plus the surfaces); `hadronize.py`
takes ~40 s per path, each comparison script a few minutes.
