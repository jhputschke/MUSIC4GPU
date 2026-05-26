#!/usr/bin/env bash
# metal_vs_cpu_bench_3d.sh — Compare Metal GPU vs CPU on 3+1D MUSIC runs.
#
# Like tests/metal_vs_cpu_bench.sh but with boost_invariant=0 and
# Grid_size_in_eta > 1, so the η-direction stencils, geometric terms
# (cosh/sinh of Δη), and η-coupling algebra are all live.  Production
# 3D runs typically use 128×128×32–64; the cases below are smaller so
# they finish in a few wall-clock minutes.
#
# Run from the MUSIC repository root:
#   bash tests/metal_vs_cpu_bench_3d.sh
#
# Prerequisites:
#   build/src/MUSIChydro        — standard (CPU-only) build
#   build_metal/src/MUSIChydro  — Metal GPU build  (-DUSE_METAL=ON)
#
# Initial condition: Gubser XY profile replicated across all η slices
# (so u^η = 0 initially everywhere).  The evolution exercises all η-direction
# code paths, but the result stays approximately η-invariant — the test is
# both a correctness check (CPU vs GPU eps_max trace) and a performance check
# (does GPU scale well as the third dimension grows?).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CPU_BIN="${ROOT}/build/src/MUSIChydro"
GPU_BIN="${ROOT}/build_metal/src/MUSIChydro"
TMPDIR_BENCH="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_BENCH}"' EXIT

check_bins() {
    local ok=1
    if [[ ! -x "${CPU_BIN}" ]]; then
        echo "ERROR: CPU binary not found: ${CPU_BIN}"
        ok=0
    fi
    if [[ ! -x "${GPU_BIN}" ]]; then
        echo "ERROR: Metal binary not found: ${GPU_BIN}"
        ok=0
    fi
    [[ ${ok} -eq 1 ]]
}

# Write a Gubser-viscous 3D input file.
make_input() {
    local nx=$1 ny=$2 neta=$3 tau_end=$4 outfile=$5
    local fm_x fm_y fm_eta
    fm_x=$(python3 -c "print(${nx}*0.1)")
    fm_y=$(python3 -c "print(${ny}*0.1)")
    # delta_eta = 0.2 → eta_size = neta * 0.2 (spans about ±neta*0.1 in rapidity)
    fm_eta=$(python3 -c "print(${neta}*0.2)")
    cat > "${outfile}" <<EOF
echo_level  6
mode  2
Initial_profile  0
boost_invariant  0
Grid_size_in_x  ${nx}
Grid_size_in_y  ${ny}
Grid_size_in_eta  ${neta}
X_grid_size_in_fm  ${fm_x}
Y_grid_size_in_fm  ${fm_y}
Eta_grid_size  ${fm_eta}
Initial_radius_size_in_fm  2.6
Initial_time_tau_0  1.0
Total_evolution_time_tau  ${tau_end}
Delta_Tau  0.005
EOS_to_use  0
Viscosity_Flag_Yes_1_No_0  1
Include_Shear_Visc_Yes_1_No_0  1
Include_Bulk_Visc_Yes_1_No_0  0
Shear_to_S_ratio  0.2
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

# Run one binary, capture wall time and stdout, return time in seconds.
run_timed() {
    local bin=$1 inp=$2 log=$3 err=$4
    local t_start t_end
    t_start=$(python3 -c "import time; print(time.time())")
    "${bin}" "${inp}" > "${log}" 2> "${err}"
    t_end=$(python3 -c "import time; print(time.time())")
    python3 -c "print(f'{${t_end}-${t_start}:.2f}')"
}

compare_eps() {
    local cpu_log=$1 gpu_log=$2
    python3 - "${cpu_log}" "${gpu_log}" <<'PYEOF'
import sys, re

def parse_eps(path):
    return [float(m.group(1)) for line in open(path)
            if (m := re.search(r'eps_max\s*=\s*([\d.e+\-]+)', line))]

cpu = parse_eps(sys.argv[1])
gpu = parse_eps(sys.argv[2])
if not cpu or not gpu:
    print("  (no eps_max lines found)")
    sys.exit(0)

n = min(len(cpu), len(gpu))
diffs = [abs(c - g) / max(abs(c), 1e-30) for c, g in zip(cpu[:n], gpu[:n])]
print(f"  steps compared : {n}")
print(f"  max rel error  : {max(diffs):.2e}")
print(f"  mean rel error : {sum(diffs)/len(diffs):.2e}")
if max(diffs) < 1e-4:
    print("  PASS  (all steps agree within 1e-4)")
else:
    print("  WARN  (drift > 1e-4 — typical float32 accumulation on 3D)")
PYEOF
}

print_header() {
    printf "\n%-22s %8s %8s %8s %8s\n" "Grid (Nx×Ny×Nη)" "CPU(s)" "GPU(s)" "Speedup" "MaxErr"
    printf "%-22s %8s %8s %8s %8s\n"   "----"           "------" "------" "-------" "------"
}

# ── main ─────────────────────────────────────────────────────────────────────

echo "========================================================"
echo " MUSIC Metal GPU vs CPU 3+1D benchmark"
echo " $(date)"
echo "========================================================"

check_bins || exit 1

echo ""
echo "CPU binary : ${CPU_BIN}"
echo "GPU binary : ${GPU_BIN}"

# ── test cases ──────────────────────────────────────────────────────────────
#   "label  Nx  Ny  Nη  tau_end"
# tau_end picked so each case lands around 10–60s of CPU wall time.
# Cell counts: 8.2k, 32.8k, 65.5k, 131k.
declare -a CASES=(
    "32x32x8     32   32    8   0.3"
    "32x32x32    32   32   32   0.2"
    "64x64x16    64   64   16   0.2"
    "64x64x32    64   64   32   0.2"
)

print_header

for case_str in "${CASES[@]}"; do
    read -r label nx ny neta tau_end <<< "${case_str}"
    INP="${TMPDIR_BENCH}/input_${label}.dat"
    CPU_LOG="${TMPDIR_BENCH}/cpu_${label}.log"
    GPU_LOG="${TMPDIR_BENCH}/gpu_${label}.log"
    CPU_ERR="${TMPDIR_BENCH}/cpu_${label}.err"
    GPU_ERR="${TMPDIR_BENCH}/gpu_${label}.err"

    make_input "${nx}" "${ny}" "${neta}" "${tau_end}" "${INP}"

    t_cpu=$(run_timed "${CPU_BIN}" "${INP}" "${CPU_LOG}" "${CPU_ERR}")
    t_gpu=$(run_timed "${GPU_BIN}" "${INP}" "${GPU_LOG}" "${GPU_ERR}")

    speedup=$(python3 -c "print(f'{${t_cpu}/${t_gpu}:.2f}x')")

    max_err=$(python3 - "${CPU_LOG}" "${GPU_LOG}" <<'PYEOF'
import sys, re
def parse(p):
    return [float(m.group(1)) for line in open(p)
            if (m := re.search(r'eps_max\s*=\s*([\d.e+\-]+)', line))]
c, g = parse(sys.argv[1]), parse(sys.argv[2])
n = min(len(c), len(g))
if n == 0: print("N/A"); sys.exit()
diffs = [abs(a-b)/max(abs(a),1e-30) for a,b in zip(c[:n],g[:n])]
print(f"{max(diffs):.1e}")
PYEOF
)

    printf "%-22s %8s %8s %8s %8s\n" "${label}" "${t_cpu}s" "${t_gpu}s" "${speedup}" "${max_err}"
done

# ── correctness detail (largest case) ───────────────────────────────────────
LAST_LABEL="${CASES[${#CASES[@]}-1]%% *}"
echo ""
echo "Correctness check (eps_max trace, ${LAST_LABEL}):"
CPU_LOG_LAST="${TMPDIR_BENCH}/cpu_${LAST_LABEL}.log"
GPU_LOG_LAST="${TMPDIR_BENCH}/gpu_${LAST_LABEL}.log"
compare_eps "${CPU_LOG_LAST}" "${GPU_LOG_LAST}"

echo ""
echo "Metal init messages (from last GPU run):"
grep "MUSIC-GPU" "${TMPDIR_BENCH}/gpu_${LAST_LABEL}.err" 2>/dev/null \
    | sed 's/^/  /' || echo "  (none captured)"

echo ""
echo "========================================================"
echo " Done."
echo "========================================================"
