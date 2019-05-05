#!/bin/bash

# TC config 
pop_size=1
generations=1
logfile="log.txt"

run_tc() {
    echo $1 $2 $3 $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $4 &>> $5
    echo -e "\n\n\n\n" >> $4
    return
}

rm $logfile

sizes1=('50 50 50')
for program in matmul map tmm tbmm; do
    for size in ${sizes1[*]}; do
        run_tc $pop_size $generations $program $size $logfile
    done
done

for size in "'256 256 512 14 3'"; do
    run_tc $pop_size $generations conv2d $size $logfile
done

#TVM config 