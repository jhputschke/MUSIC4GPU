# Latent heap OOB in the MUSIC GPU / freeze-out path

> **Status:** characterized, deterministic repro in hand, **not yet root-caused**.
> Next step is an AddressSanitizer run (details below). Discovered 2026-05-29
> while building Phase 2b (`AddGPUImprovements.md`); the cause is **pre-existing**
> in the GPU port, not introduced by Phase 1 or 2b.

## Symptom

A run that is otherwise correct segfaults at the very end, in
`MpiMusic::PassHydroSurfaceToFramework()` (libJetScape), while reading the
freeze-out surface. `Evolve::get_number_of_surface_cells()` (=
`surfaceCellVec_.size()`, `evolve.h:37,49`) returns a garbage count instead of
the physical ~97477:

```
Total number of MUSIC fluid cells: 98400000      <- correct
Total number of MUSIC surface cells: 36165927565175999   <- garbage (~2^55), then *** Break *** segmentation violation
```

Observed garbage counts across runs: `36165927565175999`, `36165919233621663`,
`36165986418452639`, `36166014875729599` — all ≈ 3.6166e16 ≈ 2^55, high bits
`0x80…`, low bits varying. This is a **corrupted `std::vector` control block**
(the `_M_finish`/`_M_start` pointers), not 2^55 real elements (the run finishes
in normal time). The fluid-cell evolution output and the hydro evolution itself
are **correct** — only the freeze-out surface vector is clobbered.

## Deterministic repro

The corruption is triggered by **any change to `sizeof(GPUGrid)`**:

| Build | `sizeof(GPUGrid)` vs baseline | Surface result |
|-------|------------------------------|----------------|
| Committed baseline (music4gpu `d64e102` / `2a55f96`) | unchanged | ✅ 97477, clean finish |
| `evo_pack_out` member added to `GPUGrid` + `buf_handles_[32]→[40]`, kernel **on** | larger | ✗ garbage, segfault |
| …same, pack kernel **disabled** | larger | ✗ segfault |
| …same, `evo_pack_out` **allocation** disabled (member still present) | larger | ✗ segfault |

Run harness (RTX 3090, discrete; `g_cuda_coherent = false`):
```
cd build_gpu && OMP_NUM_THREADS=16 ./runJetscape OO_one_event.xml
```
Healthy run prints `surface cells: 97477` and `JetScape finished after 1 events!`.
~30–45 s. (Config: MCGlauber → MUSIC, EOS 91, 100×100×60 grid,
`output_evolution_to_memory=1`, fixed `Random seed 42`.)

To re-create a crashing binary from the clean tree: add any member to `GPUGrid`
(e.g. `char _pad[64];`) and rebuild `MUSIChydro`.

## Why a GPUGrid size change does it (object layout)

`gpu_grid_` (a `GPUGrid`) is a member of `Advance`, which is a member of
`Evolve`, declared *before* `surfaceCellVec_`:

```
class Evolve {                       // evolve.h
    const EOS &eos;
    InitData &DATA;
    shared_ptr<HydroSourceBase> ...;
    Cell_info  grid_info;
    Advance    advance;              // <-- contains GPUGrid gpu_grid_
    pretty_ostream music_message;
    int rk_order;
    vector<double> epsFO_list;
    vector<SurfaceCell> surfaceCellVec_;   // evolve.h:37  <-- corrupted
    vector<double> FO_nBvsEta_, FO_nQvsEta_, FO_nSvsEta_;
};
```

`Evolve` is heap-allocated (one `new Evolve`). Growing `GPUGrid` shifts every
member after `advance` — including `surfaceCellVec_` — to a higher offset within
the `Evolve` block. So there is an **out-of-bounds write that always happens and
is data-independent**, but whose *victim* depends on layout: in the baseline
layout it lands on something benign; once `surfaceCellVec_` moves into its path,
it clobbers the vector's control block.

## What we know / have ruled out

- **Data-independent, always-executes.** The clean baseline runs the *same* GPU
  evolution (identical `e/u`, verified to 1e-7) and never crashes, so the bad
  write is not triggered by the freeze-out data — it fires every run and only
  lands fatally when the layout shifts. So it is a fixed / off-by-one OOB write,
  not a degenerate-cell condition.
- **Not the pack kernel / not Phase 2b code.** Crash reproduces with the pack
  kernel disabled and with its allocation disabled — i.e. with only the
  `GPUGrid` size change present. Phase 2b therefore keeps its scratch buffer on
  the `CUDAPipelines` singleton (not on `GPUGrid`) so `Evolve`'s layout stays
  byte-identical and the bug stays dormant. **That is a workaround, not a fix.**
- **Ruled out: `GPUGrid::buf_handles_` overflow.** Only ~29–30 of 32 slots are
  used on the discrete path (3 snapshots ×5 device bufs = 15, +8 scratch, +2
  reduce, +4 EOS); bumping it to `[40]` still crashed, so the array doesn't
  overflow — it's the layout shift that matters.
- **A separate, real latent OOB found (probably not THE active one):**
  `Polygon::add_line` (`cornelius.cpp:442`) and `Polyhedron::add_polygon`
  (`cornelius.cpp:748`) do `lines[Nlines++] = l` / `polygons[Npolygons++] = p`
  with **no bounds check** on the `donotcheck || N==0` path, into heap arrays of
  fixed size `MAX_LINES = MAX_POLYGONS = 24` (`cornelius.h:73,100`). This
  overflows if a cell ever yields >24 lines/polygons. Baseline-clean with this
  data suggests it isn't the active corruptor here, but it should be fixed
  regardless (add `if (N >= MAX_…) return false;`). It corrupts the heap (not the
  in-object vector deterministically), so it doesn't match the signature above.

## Next step: AddressSanitizer (recommended)

ASan flags the OOB **write itself**, independent of where the victim lands — so
it works on the **clean** build (`2a55f96`); no need to reproduce the crash. The
corrupted object is a host `std::vector`, so host ASan is sufficient (device
code need not be instrumented).

Suggested setup (separate build dir to keep the normal build intact):
```bash
cmake -S . -B build_asan \
  -DUSE_CUDA=ON -DUSE_MUSIC=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
  -DCMAKE_CUDA_FLAGS="-Xcompiler -fsanitize=address -Xcompiler -fno-omit-frame-pointer"
cmake --build build_asan -j   # heavy: rebuilds music4gpu (+ JetScape if full tree)

# CUDA reserves ASan's shadow-gap region, so disable that guard; skip leak
# reports (the tree leaks elsewhere — we only want the OOB).
ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=0:halt_on_error=1:abort_on_error=1" \
  OMP_NUM_THREADS=4 ./runJetscape OO_one_event.xml 2>&1 | tee /tmp/asan.log
```
Expect a `heap-buffer-overflow`/`stack-buffer-overflow`/`global-buffer-overflow`
**WRITE** report with the culprit stack (in `evolve.cpp` / `cornelius.cpp` /
`grid_info.cpp` / the GPU host code). That is the bug.

Caveats / gotchas:
- ASan + CUDA usually needs `protect_shadow_gap=0` (above), else it aborts at
  CUDA init.
- Prebuilt libs (ROOT/SMASH/Pythia) are not instrumented; ASan still intercepts
  global `malloc/free` so **heap** OOB anywhere is caught, but stack/global OOB
  inside non-instrumented libs is not. The freeze-out code we care about is in
  `libmusic`, which is instrumented here.
- If the full-tree ASan build is too entangled, a cheaper route is a standalone
  `MUSIChydro` ASan build driven on an initial condition that produces a
  freeze-out surface (the finder is shared).

## Alternative: gdb hardware watchpoint

On a *crashing-layout* binary (add a dummy `GPUGrid` member), no sanitizer needed:
```
gdb --args ./runJetscape OO_one_event.xml
break Evolve::EvolveOneTimeStep
run
# at first stop:
p &this->surfaceCellVec_          # ADDR; _M_finish is at ADDR+8
watch *(unsigned long*)(ADDR+8) if *(unsigned long*)(ADDR+8) > 0x7fffffffffffUL
continue                          # legit push_backs stay < 0x7fff…; the
                                  # corrupting write trips the condition -> bt
```
The condition filters out the legitimate `push_back`/realloc writes (valid heap
pointers) and stops only on the wild value.

## Also worth doing regardless

- Bounds-check `Polygon::add_line` / `Polyhedron::add_polygon` (`cornelius.cpp:442,748`).
- Once root-caused, re-test by putting Phase 2b's `evo_pack_out` back onto
  `GPUGrid` (simpler ownership) and confirming the surface stays 97477.

## Pointers

- Repro/profile harness and `MUSIC_PROFILE=1` notes: this file's run command;
  see also the project memory `music-gpu-test-harness`.
- Phase 2b context and the layout-safe workaround: `AddGPUImprovements.md`
  (§"As built (CUDA) and measured").
- music4gpu is a **nested git repo**; this work is on branch `AddGPUTuning`.
