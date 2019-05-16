#!/bin/bash

# - - - TC config - - - 
pop_size=100
generations=12
tc_logfile="tc_log.txt"

mm_arg1='72 26 26'
mm_arg2='50 50 50'
mm_arg3='128 32 256'
mm_arg4='128 1024 1024'
mm_arg5='128 4096 16384'

tbmm_arg1='72 26 26 250'
tbmm_arg2='72 26 26 500'
tbmm_arg3='72 26 26 1024'
tbmm_arg4='50 50 50 500'
tbmm_arg5='50 50 50 1024'

conv_arg1='32 16 16 14 14'
conv_arg2='32 32 32 7 7'
conv_arg3='32 4 4 56 56'
conv_arg4='32 8 8 28 28'
conv_arg5='256 256 512 14 3'

map_arg1='1000'
map_arg2='10000'
map_arg3='100000'
map_arg4='1000000'
map_arg5='10000000'

run_tc() {
    echo $1 $2 $3 $4

    if [ "$3" = "matmul" ] || [ "$3" = "tmm" ]; then
        arg1=$mm_arg1
        arg2=$mm_arg2 
        arg3=$mm_arg3 
        arg4=$mm_arg4 
        arg5=$mm_arg5 
    elif [ "$3" = "tbmm" ]; then
        arg1=$tbmm_arg1
        arg2=$tbmm_arg2 
        arg3=$tbmm_arg3 
        arg4=$tbmm_arg4 
        arg5=$tbmm_arg5 
    elif [ "$3" = "conv2d" ]; then
        arg1=$conv_arg1
        arg2=$conv_arg2
        arg3=$conv_arg3
        arg4=$conv_arg4
        arg5=$conv_arg5
    else # [ "$3" = "map" ] 
        arg1=$map_arg1
        arg2=$map_arg2
        arg3=$map_arg3
        arg4=$map_arg4
        arg5=$map_arg5
    fi 

    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg1 &>> $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg1 &>> $4

    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg2 &>> $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg2 &>> $4

    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg3 &>> $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg3 &>> $4

    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg4 &>> $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg4 &>> $4

    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --prog=$3 --size $arg5 &>> $4
    /opt/conda/anaconda/envs/tc_build/bin/python3 tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$1 --tuner_generations=$2 --prog=$3 --size $arg5 &>> $4

    echo -e "\n - - - - \n" >> $4
    return
}

rm $tc_logfile

for program in matmul tmm tbmm conv2d  map; do
    run_tc $pop_size $generations $program $tc_logfile    
done

# - - - TVM config - - - 
tvm_logfile="tvm_log.txt"

run_tvm() { 
    echo $1 $2

    if [ "$1" = "matmul" ] || [ "$1" = "tmm" ]; then
        arg1=$mm_arg1
        arg2=$mm_arg2 
        arg3=$mm_arg3 
        arg4=$mm_arg4 
        arg5=$mm_arg5 
    elif [ "$1" = "tbmm" ]; then
        arg1=$tbmm_arg1
        arg2=$tbmm_arg2 
        arg3=$tbmm_arg3 
        arg4=$tbmm_arg4 
        arg5=$tbmm_arg5 
    elif [ "$1" = "conv2d" ]; then
        arg1=$conv_arg1
        arg2=$conv_arg2
        arg3=$conv_arg3
        arg4=$conv_arg4
        arg5=$conv_arg5
    else # [ "$1" = "map" ] 
        arg1=$map_arg1
        arg2=$map_arg2
        arg3=$map_arg3
        arg4=$map_arg4
        arg5=$map_arg5
    fi 

    python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg1 >> $2
    python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg2 >> $2
    python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg3 >> $2
    python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg4 >> $2
    python tvm/tvm_bench.py --debug=True --prog=$1 --size $arg5 >> $2

    echo -e "\n - - - - \n" >> $2
    return
}

rm $tvm_logfile

for program in matmul tmm tbmm conv2d map; do
    run_tvm $program $tvm_logfile    
done

git add . && git commit -m 'Refactoring' && git push origin