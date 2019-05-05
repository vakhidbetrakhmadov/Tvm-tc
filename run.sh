#!/bin/bash

# - - - TC config - - - 

pop_size=1
generations=1
tc_logfile="tc_log.txt"

run_tc() {
    echo $1 $2 $3 $4

    arg1_1='256 256 512 14 3'
    arg2_1='50 50 50 50'
    
    if [ "$3" = "conv2d" ]; then
        /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg1_1 &>> $4
        /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg1_1 &>> $4
    else 
        /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg2_1 &>> $4
        /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg2_1 &>> $4
    fi 

    echo -e "\n - - - - \n" >> $4
    return
}

# rm $tc_logfile

# for program in matmul map conv2d tmm tbmm; do
#     run_tc $pop_size $generations $program $tc_logfile    
# done

# - - - TVM config - - - 

tvm_logfile="tvm_log.txt"

run_tvm() { 
    echo $1 $2

    arg1_1='256 256 512 14 3'
    arg2_1='50 50 50 50'

    if [ "$1" = "conv2d" ]; then
        python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg1_1 >> $2
    else 
        python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg2_1 >> $2
    fi 

    echo -e "\n - - - - \n" >> $2
    return
}

rm $tvm_logfile

for program in matmul map conv2d tmm tbmm; do
    run_tvm $program $tvm_logfile    
done