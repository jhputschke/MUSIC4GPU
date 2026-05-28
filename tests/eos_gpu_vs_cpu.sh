#!/usr/bin/env bash
# eos_gpu_vs_cpu.sh — Verify a given EOS runs correctly on the GPU EOS path.
#
# Primary purpose: confirm that the GPU's pre-sampled EOS tables (P, dP/de,
# s, T — log-spaced; see PORT_GPU.md §4.4) reproduce the CPU EOS for a
# NON-conformal equation of state.  Defaults to EOS 91 (zero-muB hotQCD,
# SMASH variant), the case that exposed the linear-vs-log table-sampling bug.
#
# Works on any hardware: auto-detects a Metal or CUDA GPU build.  The
# "[MUSIC-GPU] GPU grid allocated" / "configuration outside GPU support
# matrix" log lines are backend-agnostic, so the same script verifies both.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   Run from the MUSIC repository root:
#       bash tests/eos_gpu_vs_cpu.sh
#
#   Override the EOS or grid via environment variables:
#       EOS_ID=9  bash tests/eos_gpu_vs_cpu.sh      # standard hotQCD
#       EOS_ID=0  TOL=1e-4 bash tests/eos_gpu_vs_cpu.sh   # ideal gas (tight)
#       GRID=64 TAU_END=0.5 bash tests/eos_gpu_vs_cpu.sh
#
#   Point at specific binaries (otherwise auto-detected):
#       CPU_BIN=/path/MUSIChydro GPU_BIN=/path/MUSIChydro bash tests/eos_gpu_vs_cpu.sh
#
# ── Prerequisites ─────────────────────────────────────────────────────────────
#   CPU build:    cmake -S . -B build && cmake --build build -j
#   Metal build:  cmake -S . -B build_metal -DUSE_METAL=ON && cmake --build build_metal -j
#   CUDA build:   cmake -S . -B build_cuda  -DUSE_CUDA=ON  && cmake --build build_cuda  -j
#
#   EOS tables (table-based EOS only) must be present under EOS/.  For the
#   default EOS 91 the script checks for and, if missing, tells you to run:
#       (cd EOS && bash download_hotQCD.sh SMASH_binary)
#
# ── What it checks (all must pass) ────────────────────────────────────────────
#   1. The GPU build actually dispatched to the GPU EOS path (not CPU fallback).
#   2. The GPU run completed without the "energy density increased by >5x"
#      sanity abort (the failure signature of the old table-sampling bug).
#   3. The GPU eps_max(tau) trace agrees with the CPU within TOL.
#
# Note: with Gubser initial conditions (Initial_profile 0) the analytic
# "Autocheck" comparison printed by MUSIC is only meaningful for the
# conformal ideal gas (EOS 0).  For EOS != 0 this is purely a GPU-vs-CPU
# *agreement* test — both binaries solve the identical PDE, so they must
# match regardless of whether the IC is a physical solution of that EOS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# EOS resolves its tables relative to $HYDROPROGRAMPATH (default ".").  Pin it
# to the repo root so the script works from any working directory.
export HYDROPROGRAMPATH="${ROOT}"

# ── tunables (env-overridable) ────────────────────────────────────────────────
EOS_ID="${EOS_ID:-91}"
GRID="${GRID:-32}"
TAU_END="${TAU_END:-0.5}"      # tau_end - tau0; ~100 steps at Delta_Tau 0.005
TOL="${TOL:-1e-3}"             # max allowed rel err on eps_max trace.
                              #   ideal gas (EOS 0) hits ~1e-4; non-conformal
                              #   EOS (hotQCD/WB/s95p) sit at the ~1e-4..1e-3
                              #   float32 floor.  A healthy fix is << 1e-2;
                              #   the table-sampling bug gave ~6e-2 + a crash.

# ── locate binaries ───────────────────────────────────────────────────────────
CPU_BIN="${CPU_BIN:-${ROOT}/build/src/MUSIChydro}"

if [[ -z "${GPU_BIN:-}" ]]; then
    for cand in build_metal build_cuda build_gpu; do
        if [[ -x "${ROOT}/${cand}/src/MUSIChydro" ]]; then
            GPU_BIN="${ROOT}/${cand}/src/MUSIChydro"
            break
        fi
    done
fi

fail=0
if [[ ! -x "${CPU_BIN}" ]]; then
    echo "ERROR: CPU binary not found: ${CPU_BIN}"
    echo "  Build it: cmake -S . -B build && cmake --build build -j"
    fail=1
fi
if [[ -z "${GPU_BIN:-}" || ! -x "${GPU_BIN}" ]]; then
    echo "ERROR: no GPU binary found (looked for build_metal/build_cuda/build_gpu)."
    echo "  Metal: cmake -S . -B build_metal -DUSE_METAL=ON && cmake --build build_metal -j"
    echo "  CUDA : cmake -S . -B build_cuda  -DUSE_CUDA=ON  && cmake --build build_cuda  -j"
    echo "  Or set GPU_BIN=/path/to/MUSIChydro"
    fail=1
fi
[[ ${fail} -eq 1 ]] && exit 1

# ── check EOS tables for table-based EOS ──────────────────────────────────────
# Maps the EOS id to a representative table file and the command that fetches it.
check_eos_tables() {
    local id="$1" file hint
    case "${id}" in
        0)  return 0 ;;  # ideal gas — analytic, no table
        9)  file="EOS/hotQCD/hrg_hotqcd_eos_binary.dat"
            hint="(cd EOS && bash download_hotQCD.sh binary)" ;;
        91) file="EOS/hotQCD/hrg_hotqcd_eos_SMASH_binary.dat"
            hint="(cd EOS && bash download_hotQCD.sh SMASH_binary)" ;;
        2|3|4|5|6|7)
            file="EOS/s95p-v1.2/s95p-v1.2_par1.dat"
            hint="(cd EOS && bash download_s95p.sh)" ;;
        *)  echo "  (no table-presence check wired for EOS ${id}; assuming present)"
            return 0 ;;
    esac
    if [[ ! -f "${ROOT}/${file}" ]]; then
        echo "ERROR: EOS ${id} needs a table that is missing:"
        echo "    ${file}"
        echo "  Download it with:"
        echo "    ${hint}"
        return 1
    fi
    return 0
}

# ── work dir ──────────────────────────────────────────────────────────────────
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

INP="${WORK}/input.dat"
write_input() {
    local fm; fm=$(python3 -c "print(${GRID}*0.1)")
    cat > "${INP}" <<EOF
echo_level  6
mode  2
Initial_profile  0
boost_invariant  1
Grid_size_in_x  ${GRID}
Grid_size_in_y  ${GRID}
Grid_size_in_eta  1
X_grid_size_in_fm  ${fm}
Y_grid_size_in_fm  ${fm}
Eta_grid_size  0.1
Initial_radius_size_in_fm  2.6
Initial_time_tau_0  1.0
Total_evolution_time_tau  ${TAU_END}
Delta_Tau  0.005
EOS_to_use  ${EOS_ID}
Viscosity_Flag_Yes_1_No_0  1
Include_Shear_Visc_Yes_1_No_0  1
Include_Bulk_Visc_Yes_1_No_0  0
Shear_to_S_ratio  0.2
Shear_relaxation_time_tau_pi  0.01
Include_Rhob_Yes_1_No_0  0
turn_on_baryon_diffusion  0
Runge_Kutta_order  2
reconst_type  1
Minmod_Theta  1.8
UseCFL_condition  0
output_evolution_data  0
Do_FreezeOut_Yes_1_No_0  0
outputBinaryEvolution  0
EndOfData
EOF
}

# ── main ──────────────────────────────────────────────────────────────────────
echo "========================================================"
echo " MUSIC GPU-vs-CPU EOS verification"
echo " EOS_to_use=${EOS_ID}  grid=${GRID}x${GRID}x1  tau_end=${TAU_END}  tol=${TOL}"
echo " $(date)"
echo "========================================================"
echo "CPU binary : ${CPU_BIN}"
echo "GPU binary : ${GPU_BIN}"

check_eos_tables "${EOS_ID}" || exit 1
write_input

CPU_LOG="${WORK}/cpu.log"; GPU_LOG="${WORK}/gpu.log"

echo
echo "Running CPU ..."
set +e
"${CPU_BIN}" "${INP}" > "${CPU_LOG}" 2>&1; cpu_rc=$?
echo "Running GPU ..."
"${GPU_BIN}" "${INP}" > "${GPU_LOG}" 2>&1; gpu_rc=$?
set -e

# ── assertions ────────────────────────────────────────────────────────────────
ok=1

# (1) GPU path actually taken?
if grep -q "configuration outside GPU support matrix" "${GPU_LOG}"; then
    echo "FAIL [1/3]: GPU fell back to CPU (config rejected by gpu_features_supported)."
    ok=0
elif grep -q "GPU grid allocated" "${GPU_LOG}"; then
    echo "PASS [1/3]: GPU EOS path dispatched ('GPU grid allocated')."
else
    echo "FAIL [1/3]: no GPU dispatch detected (is GPU_BIN a GPU build?)."
    echo "            (need echo_level >= 1 to see the [MUSIC-GPU] log lines.)"
    ok=0
fi

# (2) GPU completed without the >5x energy-density abort?
if [[ ${gpu_rc} -ne 0 ]] || grep -qi "increased by more than\|factor of 5" "${GPU_LOG}"; then
    echo "FAIL [2/3]: GPU run did not complete cleanly (rc=${gpu_rc}); checking for sanity abort:"
    grep -i "increased by more than\|factor of 5" "${GPU_LOG}" | head -1 | sed 's/^/            /' || true
    ok=0
else
    echo "PASS [2/3]: GPU run completed (rc=0, no energy-density blow-up)."
fi
[[ ${cpu_rc} -ne 0 ]] && echo "  WARN: CPU run rc=${cpu_rc} (reference may be incomplete)."

# (3) eps_max(tau) agreement within TOL
echo "Comparing eps_max(tau) traces ..."
python3 - "${CPU_LOG}" "${GPU_LOG}" "${TOL}" <<'PYEOF'
import sys, re
def parse(p):
    return [float(m.group(1)) for line in open(p)
            if (m := re.search(r'eps_max\s*=\s*([\d.eE+\-]+)', line))]
cpu, gpu, tol = parse(sys.argv[1]), parse(sys.argv[2]), float(sys.argv[3])
if not cpu or not gpu:
    print("FAIL [3/3]: no eps_max lines parsed (need echo_level >= 6).")
    sys.exit(2)
n = min(len(cpu), len(gpu))
diffs = [abs(c - g) / max(abs(c), 1e-30) for c, g in zip(cpu[:n], gpu[:n])]
mx = max(diffs); im = diffs.index(mx)
print(f"  steps compared : {n}  (CPU {len(cpu)}, GPU {len(gpu)})")
print(f"  max rel error  : {mx:.2e}  at step {im} (CPU={cpu[im]:.6g} GPU={gpu[im]:.6g})")
print(f"  mean rel error : {sum(diffs)/len(diffs):.2e}")
if len(cpu) != len(gpu):
    print(f"  WARN: trace lengths differ — one run stopped early.")
if mx < tol:
    print(f"PASS [3/3]: agrees within {tol:g}.")
    sys.exit(0)
print(f"FAIL [3/3]: max rel error {mx:.2e} exceeds tol {tol:g}.")
sys.exit(3)
PYEOF
cmp_rc=$?
[[ ${cmp_rc} -ne 0 ]] && ok=0

echo "--------------------------------------------------------"
if [[ ${ok} -eq 1 ]]; then
    echo "RESULT: PASS — EOS ${EOS_ID} runs correctly on the GPU EOS path."
    exit 0
else
    echo "RESULT: FAIL — see messages above."
    exit 1
fi
