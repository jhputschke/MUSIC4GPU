#!/usr/bin/env bash
# cuda_vs_cpu_bench.sh — Compare CUDA GPU vs CPU builds of MUSIC (3+1D production).
#
# Run from the MUSIC repository root:
#   OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench_3d.sh
#
# Prerequisites:
#   build/src/MUSIChydro       — standard (CPU-only) build
#   build_cuda/src/MUSIChydro  — CUDA GPU build  (-DUSE_CUDA=ON)
#
# All test cases use Initial_profile 0 (analytical Gubser initial conditions),
# which needs no external data files.  Accept threshold: max rel eps_max < 1e-3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CPU_BIN="${ROOT}/build/src/MUSIChydro"
GPU_BIN="${ROOT}/build_cuda/src/MUSIChydro"
TMPDIR_BENCH="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_BENCH}"' EXIT

NCORES="$(nproc 2>/dev/null || echo 4)"

check_bins() {
    local ok=1
    if [[ ! -x "${CPU_BIN}" ]]; then
        echo "ERROR: CPU binary not found: ${CPU_BIN}"
        echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j${NCORES}"
        ok=0
    fi
    if [[ ! -x "${GPU_BIN}" ]]; then
        echo "ERROR: CUDA binary not found: ${GPU_BIN}"
        echo "  cmake -S . -B build_cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build_cuda -j${NCORES}"
        ok=0
    fi
    [[ ${ok} -eq 1 ]]
}

# Write a Gubser-viscous input file with given grid dimensions (boost-invariant).
make_input() {
    local nx=$1 ny=$2 neta=$3 tau_end=$4 outfile=$5
    local fm_x fm_y fm_eta
    fm_x=$(python3 -c "print(${nx}*0.1)")
    fm_y=$(python3 -c "print(${ny}*0.1)")
    # delta_eta = 0.2 → eta_size = neta * 0.2
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
    vals = []
    for line in open(path):
        m = re.search(r'eps_max\s*=\s*([\d.eE+\-]+)', line)
        if m:
            vals.append(float(m.group(1)))
    return vals
cpu = parse_eps(sys.argv[1]); gpu = parse_eps(sys.argv[2])
if not cpu or not gpu:
    print("  (no eps_max lines found — check echo_level)"); sys.exit(0)
n = min(len(cpu), len(gpu))
diffs = [abs(c - g) / max(abs(c), 1e-30) for c, g in zip(cpu[:n], gpu[:n])]
print(f"  steps compared : {n}")
print(f"  max rel error  : {max(diffs):.2e}")
print(f"  mean rel error : {sum(diffs)/len(diffs):.2e}")
print("  PASS  (all steps agree within 1e-3)" if max(diffs) < 1e-3
      else "  FAIL  (discrepancy > 1e-3 — check kernel correctness)")
PYEOF
}

print_header() {
    printf "\n%-20s %8s %8s %8s %8s\n" "Grid" "CPU(s)" "GPU(s)" "Speedup" "MaxErr"
    printf "%-20s %8s %8s %8s %8s\n" "----" "------" "------" "-------" "------"
}

echo "========================================================"
echo " MUSIC CUDA GPU vs CPU benchmark (3+1D production)"
echo " $(date)"
echo "========================================================"

check_bins || exit 1

echo ""
echo "CPU binary : ${CPU_BIN}  (OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset})"
echo "GPU binary : ${GPU_BIN}"
command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU        : /'

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
            if (m := re.search(r'eps_max\s*=\s*([\d.eE+\-]+)', line))]
c, g = parse(sys.argv[1]), parse(sys.argv[2])
n = min(len(c), len(g))
if n == 0: print("N/A"); sys.exit()
diffs = [abs(a-b)/max(abs(a),1e-30) for a,b in zip(c[:n],g[:n])]
print(f"{max(diffs):.1e}")
PYEOF
)
    printf "%-20s %8s %8s %8s %8s\n" "${label}" "${t_cpu}s" "${t_gpu}s" "${speedup}" "${max_err}"
done

LAST_LABEL="${CASES[${#CASES[@]}-1]%% *}"
echo ""
echo "Correctness check (eps_max trace, ${LAST_LABEL}):"
compare_eps "${TMPDIR_BENCH}/cpu_${LAST_LABEL}.log" "${TMPDIR_BENCH}/gpu_${LAST_LABEL}.log"

echo ""
echo "CUDA init messages (from last GPU run):"
grep "MUSIC-GPU" "${TMPDIR_BENCH}/gpu_${LAST_LABEL}.err" 2>/dev/null | sed 's/^/  /' || echo "  (none captured)"

echo ""
echo "========================================================"
echo " Done."
echo "========================================================"
