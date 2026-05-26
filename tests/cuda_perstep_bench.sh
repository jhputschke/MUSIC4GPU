#!/usr/bin/env bash
# cuda_perstep_bench.sh — Per-step compute time, CUDA GPU vs CPU.
#
# Wall-clock totals are dominated by fixed overhead (CUDA context init + EOS
# sampling, ~0.5 s) which masks per-kernel performance on short runs.  This
# script instead runs each grid at two step counts (N0 and N0+100) and takes
# the difference, isolating the per-step cost:
#       per_step = ( T(long) - T(short) ) / 100
# Each timing is the min of 3 runs to suppress scheduler/thermal noise.
#
# Usage (from repo root):
#   OMP_NUM_THREADS=$(nproc) bash tests/cuda_perstep_bench.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CPU_BIN="${ROOT}/build/src/MUSIChydro"
GPU_BIN="${ROOT}/build_cuda/src/MUSIChydro"
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
REPEAT="${REPEAT:-3}"

mk() { # nx ny neta tau_end boost outfile
cat > "$6" <<EOF
echo_level 1
mode 2
Initial_profile 0
boost_invariant $5
Grid_size_in_x $1
Grid_size_in_y $2
Grid_size_in_eta $3
X_grid_size_in_fm $(python3 -c "print($1*0.1)")
Y_grid_size_in_fm $(python3 -c "print($2*0.1)")
Eta_grid_size $(python3 -c "print($3*0.2 if $3>1 else 0.1)")
Initial_radius_size_in_fm 2.6
Initial_time_tau_0 1.0
Total_evolution_time_tau $4
Delta_Tau 0.005
EOS_to_use 0
Viscosity_Flag_Yes_1_No_0 1
Include_Shear_Visc_Yes_1_No_0 1
Include_Bulk_Visc_Yes_1_No_0 0
Shear_to_S_ratio 0.2
Shear_relaxation_time_tau_pi 0.01
Include_Rhob_Yes_1_No_0 0
turn_on_baryon_diffusion 0
Runge_Kutta_order 2
reconst_type 1
Minmod_Theta 1.8
UseCFL_condition 0
output_evolution_data 0
Do_FreezeOut_Yes_1_No_0 0
outputBinaryEvolution 0
EndOfData
EOF
}

bestmin() { # bin input n
    local best=999 s e d
    for _ in $(seq "$3"); do
        s=$(python3 -c "import time;print(time.time())")
        "$1" "$2" >/dev/null 2>&1
        e=$(python3 -c "import time;print(time.time())")
        d=$(python3 -c "print($e-$s)")
        best=$(python3 -c "print(min($best,$d))")
    done
    echo "$best"
}

# tau for 10 steps and 110 steps at Delta_Tau=0.005
TAU_SHORT=0.05
TAU_LONG=0.55

printf "\n%-14s %5s %10s %10s %9s\n" "Grid" "boost" "GPU ms/st" "CPU ms/st" "Speedup"
printf "%-14s %5s %10s %10s %9s\n" "----" "-----" "---------" "---------" "-------"

run_case() { # label nx ny neta boost
    local label=$1 nx=$2 ny=$3 neta=$4 boost=$5
    mk "$nx" "$ny" "$neta" "$TAU_SHORT" "$boost" "${TMP}/s.dat"
    mk "$nx" "$ny" "$neta" "$TAU_LONG"  "$boost" "${TMP}/l.dat"
    local gs gl cs cl
    gs=$(bestmin "$GPU_BIN" "${TMP}/s.dat" "$REPEAT")
    gl=$(bestmin "$GPU_BIN" "${TMP}/l.dat" "$REPEAT")
    cs=$(bestmin "$CPU_BIN" "${TMP}/s.dat" "$REPEAT")
    cl=$(bestmin "$CPU_BIN" "${TMP}/l.dat" "$REPEAT")
    python3 -c "
gp=($gl-$gs)/100*1000; cp=($cl-$cs)/100*1000
print(f'{\"$label\":<14} {$boost:>5} {gp:>10.2f} {cp:>10.2f} {cp/gp:>8.2f}x')"
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
run_case "128x128x1"  128 128  1  1
run_case "64x64x16"   64  64  16  0
run_case "64x64x32"   64  64  32  0
echo ""
echo "GPU: $(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || echo CUDA)  |  CPU threads: ${OMP_NUM_THREADS}"
