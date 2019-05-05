#!/bin/bash

# TC config 
alias tc_python = '/opt/conda/anaconda/envs/tc_build/bin/python3'
logfile = "log.txt"
pop_size = 1
generations = 1

$tc_python tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$pop_size --tuner_generations=$generations --prog=matmul >> $logfile
$tc_python tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$pop_size --tuner_generations=$generations --prog=map >> $logfile
$tc_python tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$pop_size --tuner_generations=$generations --prog=conv2d >> $logfile
$tc_python tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$pop_size --tuner_generations=$generations --prog=tmm >> $logfile
$tc_python tc/tc_bench.py --debug=True --autotuner=True --store_to_cache=True --tuner_pop_size=$pop_size --tuner_generations=$generations --prog=tbmm >> $logfile