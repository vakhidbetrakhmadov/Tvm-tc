#!/bin/bash

# TC config 
pop_size=1
generations=1
logfile="log.txt"

run_tc() {
    echo $1 $2 $3 $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 &>> $4
    echo -e "\n\n\n\n" >> $4
    return
}

rm $logfile

for program in matmul map conv2d tmm tbmm
do
    run_tc $pop_size $generations $program $logfile
done