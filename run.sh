#!/bin/bash

# TC config 
pop_size=1
generations=1
logfile="log.txt"

run_tc() {
    echo $1 $2 $3 $4

    arg1='50 50 50 50'
    arg2='256 256 512 14 3'

    if [ "$3" = "conv2d" ]; then
        /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg2 &>> $4
    else 
        /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg1 &>> $4
    fi 

    echo -e "\n - - - - \n" >> $4
    return
}

rm $logfile

for program in matmul map conv2d tmm tbmm; do
    run_tc $pop_size $generations $program $logfile    
done

#TVM config 