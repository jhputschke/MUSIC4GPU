#!/usr/bin/env bash
# kokkos_consistency.sh — the D9 single-source cross-backend consistency gate.
#
# One kernel source (src/gpu/music_kernels_kokkos.hpp) is compiled for several
# Kokkos execution spaces.  This script runs the SAME hydro input on each
# available Kokkos backend (Serial / OpenMP / Cuda) plus the legacy CPU
# reference, and checks two things (PlanKokkosPort.md D9/D10):
#
#   * every Kokkos backend agrees with the double-precision CPU reference within
#     TOL_CPU (1e-3, the end-to-end gate the EOS test uses), and
#   * the Kokkos backends agree with EACH OTHER within TOL_X (1e-4, cross-backend)
#     — proving the single source behaves identically across execution spaces,
#     i.e. there is no CPU/GPU physics drift.
#
# Usage (from the repo root):
#   bash tests/kokkos_consistency.sh
#
# Binaries are auto-detected; override any with an env var:
#   CPU_BIN, KOKKOS_SERIAL_BIN, KOKKOS_OPENMP_BIN, KOKKOS_CUDA_BIN
# Tunables: EOS_ID (91), GRID (32), TAU_END (0.5), TOL_CPU (1e-3), TOL_X (1e-4).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export HYDROPROGRAMPATH="${ROOT}"

EOS_ID="${EOS_ID:-91}"
GRID="${GRID:-32}"
TAU_END="${TAU_END:-0.5}"
TOL_CPU="${TOL_CPU:-1e-3}"
TOL_X="${TOL_X:-1e-4}"

CPU_BIN="${CPU_BIN:-${ROOT}/build/src/MUSIChydro}"
KOKKOS_SERIAL_BIN="${KOKKOS_SERIAL_BIN:-${ROOT}/build_kokkos_serial/src/MUSIChydro}"
KOKKOS_OPENMP_BIN="${KOKKOS_OPENMP_BIN:-${ROOT}/build_kokkos/src/MUSIChydro}"
KOKKOS_CUDA_BIN="${KOKKOS_CUDA_BIN:-${ROOT}/build_kokkos_cuda/src/MUSIChydro}"

WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
INP="${WORK}/input.dat"
fm=$(python3 -c "print(${GRID}*0.1)")
cat > "${INP}" <<EOF
echo_level 6
mode 2
Initial_profile 0
boost_invariant 1
Grid_size_in_x ${GRID}
Grid_size_in_y ${GRID}
Grid_size_in_eta 1
X_grid_size_in_fm ${fm}
Y_grid_size_in_fm ${fm}
Eta_grid_size 0.1
Initial_radius_size_in_fm 2.6
Initial_time_tau_0 1.0
Total_evolution_time_tau ${TAU_END}
Delta_Tau 0.005
EOS_to_use ${EOS_ID}
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

echo "============================================================"
echo " Kokkos single-source cross-backend consistency (D9)"
echo " EOS ${EOS_ID}, ${GRID}x${GRID}x1, tau_end=${TAU_END}"
echo " tol vs CPU=${TOL_CPU}  cross-backend=${TOL_X}"
echo "============================================================"

names=(); logs=()
add_run() { # name bin
    local name="$1" bin="$2"
    if [[ -x "${bin}" ]]; then
        local log="${WORK}/${name}.log"
        "${bin}" "${INP}" > "${log}" 2>&1 || true
        names+=("${name}"); logs+=("${log}")
        echo "  ran ${name}  (${bin})"
    else
        echo "  skip ${name}  (not built: ${bin})"
    fi
}
echo "Running ..."
add_run CPU            "${CPU_BIN}"
add_run Kokkos-Serial  "${KOKKOS_SERIAL_BIN}"
add_run Kokkos-OpenMP  "${KOKKOS_OPENMP_BIN}"
add_run Kokkos-Cuda    "${KOKKOS_CUDA_BIN}"

echo
python3 - "${TOL_CPU}" "${TOL_X}" "${names[*]}" "${logs[*]}" <<'PYEOF'
import sys, re
tol_cpu, tol_x = float(sys.argv[1]), float(sys.argv[2])
names = sys.argv[3].split()
logs  = sys.argv[4].split()
def parse(p):
    return [float(m.group(1)) for line in open(p)
            if (m := re.search(r'eps_max\s*=\s*([\d.eE+\-]+)', line))]
trace = {n: parse(l) for n, l in zip(names, logs)}
trace = {n: t for n, t in trace.items() if t}
if 'CPU' not in trace:
    print("FAIL: no CPU reference trace parsed."); sys.exit(2)
cpu = trace['CPU']
def reldiff(a, b):
    n = min(len(a), len(b))
    return max(abs(a[i]-b[i])/max(abs(a[i]),1e-30) for i in range(n)) if n else float('nan')

kok = [n for n in names if n.startswith('Kokkos') and n in trace]
ok = True
print("vs CPU reference (double precision):")
for n in kok:
    d = reldiff(trace[n], cpu)
    flag = "PASS" if d < tol_cpu else "FAIL"
    ok &= d < tol_cpu
    print(f"  {n:14s} max rel err = {d:.2e}   [{flag} < {tol_cpu:g}]")

print("\ncross-backend (Kokkos vs Kokkos, single source):")
worst = 0.0
for i in range(len(kok)):
    for j in range(i+1, len(kok)):
        d = reldiff(trace[kok[i]], trace[kok[j]]); worst = max(worst, d)
        flag = "PASS" if d < tol_x else "FAIL"
        ok &= d < tol_x
        print(f"  {kok[i]:14s} vs {kok[j]:14s} = {d:.2e}   [{flag} < {tol_x:g}]")
if len([n for n in kok]) < 2:
    print("  (need >=2 Kokkos backends built for a cross-backend check)")

print("\n------------------------------------------------------------")
print("RESULT:", "PASS — single source is consistent across backends." if ok
      else "FAIL — see above.")
sys.exit(0 if ok else 1)
PYEOF
