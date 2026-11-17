#!/usr/bin/env bash
# Cartesian coordinates on the GPU (B2, CUDA builds), with the viscous blob of
# RunAutoTest.sh (shear, bulk, second-order terms; Initial_profile 5):
#   gpu_0, gpu_20     3 fm/c on the GPU from t0 = 0 and t0 = 20
#   gpu10, cpu10      10 steps from t0 = 20, GPU and CPU
# gpu_check.py requires that the GPU path ran, that gpu_0 and gpu_20 agree
# bit for bit (nothing in Cartesian coordinates depends on t), that gpu_20 is
# mirror symmetric under x <-> z (sigma_x = sigma_z: a Milne term left in a
# kernel breaks that, e.g. the Christoffel terms of sigma^{mu nu} by 2 in
# pi^{xy}/pi^{yz}), and that GPU and CPU agree after 10 steps in e, u^tau,
# pi^xx and Pi (terms that keep the symmetry: theta, -u^tau Pi/tau).
# Run from the MUSIC directory: about half a minute.
set -e
here=tests/test_3D_Milne_vs_Cartesian
work=test_3D_Cartesian_GPU_runs
mkdir -p $work
runs="gpu_0 gpu_20 gpu10_20 cpu10_20"
for r in $runs; do
  t=${r#*_}
  mkdir -p $work/$r
  if [ "${r#*10_}" != "$r" ]; then
    python3 $here/make_input.py cart $t 1 \
      | sed 's/^Total_evolution_time_tau .*/Total_evolution_time_tau  0.2/' \
      > $work/$r/input
  else
    python3 $here/make_input.py cart $t 1 > $work/$r/input
  fi
  ln -sfn "$PWD/EOS" $work/$r/EOS
done
for r in $runs; do
  extra=""
  [ "${r#cpu}" != "$r" ] && extra="MUSIC_FORCE_CPU=1"
  (cd $work/$r && env $extra OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} \
     MUSIC_DUMP_STATE=$PWD/state.bin ../../MUSIChydro input > run.log 2>&1)
done
cd $work
if python3 ../$here/gpu_check.py; then
  echo "All tests passed! :)"
  cd .. && rm -rf $work
else
  echo "Tests FAILED :("
  exit 1
fi
