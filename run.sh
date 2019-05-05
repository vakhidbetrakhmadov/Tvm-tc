#!/bin/bash

# TC config 
run_tc() {
    echo $1, $2 $3
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 &>> $4
    return
}

for program in matmul map conv2d tmm tbmm
do 
    run_tc 1 1 $program "log.txt"
done