import time
import logging
import sys
import numpy as np
import tvm

from tvm import autotvm
from args_parser import get_argument_parser
from matmul import matmul_parametric, matmul_autotuner

parser = get_argument_parser()
args, extra_args = parser.parse_known_args()

target_host = "llvm"
target = "cuda"

N, L, M = 50, 50, 50

if args.autotuner: 
    if args.debug: print("Autotuning schedule parameters")

    # task = autotvm.task.create(matmul_autotuner, args=(N, L, M, 'float32'), target=target)

    # # logging config (for printing tuning log to the screen)
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    # # There are two steps for measuring a config: build and run.
    # # By default, we use all cpu cores to compile program. Then measure them sequentially.
    # # We measure 5 times and take average to reduce variance.
    # measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=5))

    # # begin tuning, log records to file `matmul.log`
    # tuner = autotvm.tuner.RandomTuner(task)
    # tuner.tune(n_trial=10, measure_option=measure_option, callbacks=[autotvm.callback.log_to_file('matmul.log')])

    # # apply history best from log file
    # with autotvm.apply_history_best('matmul.log'):
    #     with tvm.target.create(target):
    #         s, arg_bufs = matmul(N, L, M, 'float32')
    #         matmul = tvm.build(s, arg_bufs)

    #         a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    #         b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    #         c_tvm = tvm.nd.empty((N, M))
    #         matmul(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

    #         print('Result: ', c_tvm)

    #         if target == "cuda" or target.startswith('opencl'):
    #             dev_module = matmul.imported_modules[0]
    #             print(dev_module.get_source())
    #         else:
    #             print(matmul.get_source())
else:
    if args.debug: print("Manual schedule parameters")

    s, arg_bufs = matmul_parametric(tvm.var("N"), tvm.var("L"), tvm.var("M"), 'float32', args)
    matmul = tvm.build(s, arg_bufs, target, target_host=target_host, name="matmul")
    
    ctx = tvm.context(target, 0)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    a_tvm = tvm.nd.array(a_np, ctx)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    b_tvm = tvm.nd.array(b_np, ctx)
    c_np = np.zeros((N,M), dtype=np.float32)
    c_tvm = tvm.nd.array(c_np, ctx)

    matmul(a_tvm, b_tvm, c_tvm)

    print('Result: ', c_tvm)

    if target == "cuda" or target.startswith('opencl'):
        dev_module = matmul.imported_modules[0]
        print(dev_module.get_source())
    else:
        print(matmul.get_source())