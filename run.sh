#!/bin/bash

# TC config 
run_tc(pop_size, generations, program) {
    /opt/conda/anaconda/envs/tc_build/bin/python3 --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=pop_size --tuner_generations=generations --prog=program >> "log.txt"
    return
}

for program in matmul, map, conv2d, tmm, tbmm
do 
    run_tc(1, 1, program)
done