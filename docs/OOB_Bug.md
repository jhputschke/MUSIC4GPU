# Latent heap OOB in the MUSIC GPU / freeze-out path

> **Status (updated 2026-05-29 follow-up):** **could NOT be reproduced** in the
> committed code (music4gpu `c7c7022`, == `AddGPUTuning`). The original
> "deterministic repro" below did not hold up under re-testing — see
> **"## 2026-05-29 follow-up"** immediately below before relying on anything in
> the original write-up. Phase 2b's pack buffer has now been moved onto `GPUGrid`
> (the change this doc predicted would crash) and it runs clean. The only concrete
> real OOB found (Cornelius `add_line`/`add_polygon`) has been bounds-guarded but
> never triggers on these inputs.

---

## 2026-05-29 follow-up — exhaustive non-reproduction + Phase 2b shipped

The crash described below **does not reproduce** in the current committed tree.
Tested on the same machine (RTX 3090, discrete, `g_cuda_coherent=false`), same
commit, identical music4gpu build flags:

| Config (all OMP=16 unless noted) | Result |
|---|---|
| build_lite, `char[64]` pad in `GPUGrid` (OMP 1 & 16) | clean 97477 |
| build_lite, scan of `char[N]` (N=8..184) **directly before `surfaceCellVec_`** | clean 97477 (every N) |
| build_lite, host-C++ **ASan** (`-fsanitize=address`) | **no** ASan error (freeze-out suppressed to 0 cells — ASan allocator artifact) |
| build_gpu (ROOT), `char[64]` pad ×8 | clean 97477 |
| build_gpu (ROOT), `char[64]` pad + **RootBulkWriter active** ×8 | clean 97477 |
| build_gpu (ROOT), **real Phase 2b** (`evo_pack_out` on `GPUGrid`) + RootBulkWriter ×8 | clean 97477 |
| build_gpu (ROOT), **doc-exact** `evo_pack_out` + `buf_handles_[32]→[40]` + RootBulkWriter ×8 | clean 97477 |
| build_gpu (ROOT), **random seed** (12 ICs, surface 32k–147k) + RootBulkWriter | clean, **0 Cornelius-guard hits** |

Notes / what this rules out or revises:
- `AddGPUTuning` and the current branch are the **same commit** (`c7c7022`), so the
  bug is not hiding in uncommitted/branch-only code.
- **The "any `GPUGrid` size change crashes" claim does not hold** — neither a 64-byte
  pad nor the actual Phase-2b member addition (nor the doc-exact `[40]` combo)
  reproduces it, with or without ROOT / RootBulkWriter / over varied initial data.
- **Forensic on the garbage count:** `size()=(_M_finish−_M_start)/sizeof(SurfaceCell=128)`.
  The observed ~3.6166e16 ⇒ `_M_finish` overwritten by ≈`0x4000…`, i.e. a
  **`double` in [2,4)** (a freeze-out coordinate/τ/γ-like value), *not* a pointer
  (low bits vary with ASLR through the still-valid `_M_start`). So if the write
  ever fires, it is a stray `double` from the freeze-out / Cornelius path.
- **Cornelius `add_line`/`add_polygon` are now bounds-guarded** (the real latent
  OOB flagged below). Instrumented runs — including 12 random ICs — show the guards
  **never fire**, so this path does not overflow for any geometry tested; it is a
  genuine-but-dormant latent bug, not the active corruptor here.
- **Phase 2b now keeps its pack buffer on `GPUGrid`** (`evo_pack_out`/`evo_pack_floats`,
  freed in `GPUGrid::release()`), replacing the `CUDAPipelines`-singleton workaround.
  Surface stays 97477; no crash.

**Conclusion:** the layout-trigger hypothesis is not supported by these tests. The
crash is either already absent in the committed code or a heisenbug not reproducible
in this environment/input. The original analysis below is retained for reference but
is **not** confirmed.

---

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
