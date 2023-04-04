#!/bin/bash

ALGO=${1:-proto}

./download.sh walker $ALGO
./download.sh quadruped $ALGO
./download.sh point_mass_maze $ALGO
./download.sh jaco $ALGO
./download.sh cheetah $ALGO
./download.sh cartpole $ALGO
