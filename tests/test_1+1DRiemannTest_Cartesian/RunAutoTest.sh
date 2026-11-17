#!/usr/bin/env bash

# CPU references (sums to 1e-8): a CUDA build would run the viscous tubes on
# the GPU (Cartesian kernels, B2), which differs in float32 at the shock
export MUSIC_FORCE_CPU=1

./MUSIChydro tests/test_1+1DRiemannTest_Cartesian/music_input_1+1DRiemannTest_z

python3 tests/test_1+1DRiemannTest_Cartesian/TestOutputFiles.py
STATUS=$?

if [ $STATUS == 0 ]; then
    echo "All tests passed! :)"
    rm -fr ./*.dat
else
    echo "Tests FAILED :("
    exit 1
fi

./MUSIChydro tests/test_1+1DRiemannTest_Cartesian/music_input_1+1DRiemannTest_y

python3 tests/test_1+1DRiemannTest_Cartesian/TestOutputFiles.py
STATUS=$?

if [ $STATUS == 0 ]; then
    echo "All tests passed! :)"
    rm -fr ./*.dat
else
    echo "Tests FAILED :("
    exit 1
fi

./MUSIChydro tests/test_1+1DRiemannTest_Cartesian/music_input_1+1DRiemannTest_x

python3 tests/test_1+1DRiemannTest_Cartesian/TestOutputFiles.py
STATUS=$?

if [ $STATUS == 0 ]; then
    echo "All tests passed! :)"
    rm -fr ./*.dat
else
    echo "Tests FAILED :("
    exit 1
fi

./MUSIChydro tests/test_1+1DRiemannTest_Cartesian/music_input_1+1DRiemannTest_z_viscous

python3 tests/test_1+1DRiemannTest_Cartesian/TestOutputVisFiles.py z
STATUS=$?

if [ $STATUS == 0 ]; then
    echo "All tests passed! :)"
    rm -fr ./*.dat
else
    echo "Tests FAILED :("
    exit 1
fi

./MUSIChydro tests/test_1+1DRiemannTest_Cartesian/music_input_1+1DRiemannTest_y_viscous

python3 tests/test_1+1DRiemannTest_Cartesian/TestOutputVisFiles.py y
STATUS=$?

if [ $STATUS == 0 ]; then
    echo "All tests passed! :)"
    rm -fr ./*.dat
else
    echo "Tests FAILED :("
    exit 1
fi

./MUSIChydro tests/test_1+1DRiemannTest_Cartesian/music_input_1+1DRiemannTest_x_viscous

python3 tests/test_1+1DRiemannTest_Cartesian/TestOutputVisFiles.py x
STATUS=$?

if [ $STATUS == 0 ]; then
    echo "All tests passed! :)"
    rm -fr ./*.dat
else
    echo "Tests FAILED :("
    exit 1
fi
