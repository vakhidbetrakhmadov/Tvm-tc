import time
import logging
import sys
import numpy as np
import tvm
import programs

from tvm import autotvm
from args_parser import get_argument_parser

parser = get_argument_parser()
args, extra_args = parser.parse_known_args()

target_host = "llvm"
target = "cuda"

def run_and_time(s, arg_bufs, name, ctx, callback):
    start = time.clock()
    exe = tvm.build(s, arg_bufs, target, target_host=target_host, name=name)
    end = time.clock()    

    print("Done compiling \"{}\" (compile time: {}ms)".format(name, (end - start) * 10 ** 3))
    
    ctx.sync()
    start = time.clock()
    output = callback(exe)
    ctx.sync()
    end = time.clock()

    print('Result: ', output)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

    if target == "cuda" or target.startswith('opencl'):
        print(exe.imported_modules[0].get_source())
    else:
        print(exe.get_source())

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
    
    if args.prog == "matmul":

        N, L, M = 50, 50, 50

        s, arg_bufs = programs.matmul_parametric(args)

        ctx = tvm.context(target, 0)
        
        a_tvm = tvm.nd.array(np.random.uniform(size=(N, L)).astype(np.float32), ctx)    
        b_tvm = tvm.nd.array(np.random.uniform(size=(L, M)).astype(np.float32), ctx)
        c_tvm = tvm.nd.array(np.zeros((N,M), dtype=np.float32), ctx)

        def callback(exe):
            exe(a_tvm, b_tvm, c_tvm)
            return c_tvm

        run_and_time(s, arg_bufs, args.prog, ctx, callback)
    
    elif args.prog == "map":

        M = 50

        s, arg_bufs = programs._map(args)

        ctx = tvm.context(target, 0)

        a_tvm = tvm.nd.array(np.random.uniform(size=(N, L)).astype(np.float32), ctx)
        b_tvm = tvm.nd.array(np.zeros((N,M), dtype=np.float32), ctx)

        def callback(exe):
            exe(a_tvm, b_tvm)
            return b_tvm
        
        run_and_time(s, arg_bufs, args.prog, ctx, callback)