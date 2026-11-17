#!/usr/bin/env bash
# 3D test of the two coordinate systems (CoorType 0, Milne; 1, Cartesian):
# a Gaussian blob at rest in the lab frame (Initial_profile 5), the same
# physical state at tau0 = t0 = 20, 40 and 80 fm, evolved for 3 fm/c, ideal
# and with shear, bulk and second-order terms; plus a Cartesian run from
# t = 0 with shear and bulk.  The two coordinate systems start on different
# surfaces (t = t0, tau = tau0, about z^2/(2 tau0) apart inside the blob), so
# their difference must fall like 1/tau0; compare.py checks that on the common
# slice z = 0 = eta and for u^z at z = +-1 fm.  CPU path (MUSIC_FORCE_CPU;
# the GPU kernels in Cartesian coordinates: RunGPUTest.sh).  Run from the
# MUSIC directory, as the other tests: about 10 minutes on 20 cores.
set -e
here=tests/test_3D_Milne_vs_Cartesian
work=test_3D_Milne_vs_Cartesian_runs
mkdir -p $work
runs=""
for v in 0 1; do for t in 20 40 80; do for c in cart milne; do
  runs="$runs ${c}_${t}_v${v}"; done; done; done
runs="$runs cart_0_v1"
for r in $runs; do
  c=${r%%_*}; rest=${r#*_}; t=${rest%%_*}; v=${rest##*_v}
  mkdir -p $work/$r
  python3 $here/make_input.py $c $t $v > $work/$r/input
  ln -sfn "$PWD/EOS" $work/$r/EOS
done
export MUSIC_FORCE_CPU=1
printf "%s\n" $runs | xargs -P ${NJOBS:-5} -I{} bash -c \
  "cd $work/{} && OMP_NUM_THREADS=${OMP_PER_JOB:-4} MUSIC_DUMP_STATE=\$PWD/state.bin ../../MUSIChydro input > run.log 2>&1"
cd $work
if python3 ../$here/compare.py; then
  echo "All tests passed! :)"
  cd .. && rm -rf $work
else
  echo "Tests FAILED :("
  exit 1
fi
