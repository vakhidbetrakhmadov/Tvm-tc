import tvm
import argparse

from tvm import autotvm

def matmul_parametric(args):
    N, L, M = tvm.var("N"), tvm.var("L"), tvm.var("M")
    A = tvm.placeholder((N, L), name='A', dtype="float32")
    B = tvm.placeholder((L, M), name='B', dtype="float32")

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    x, y = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    xo, xi = s[C].split(x, args.x)
    yo, yi = s[C].split(y, args.y)
    
    # s[C].reorder(xo, yo, k, xi, yi)

    s[C].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[C].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[C].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[C].bind(yi, tvm.thread_axis("threadIdx.y"))

    return s, [A, B, C]

def _map(args):
    M = tvm.var("M")
    A = tvm.placeholder((M), name='A', dtype="float32")

    B = tvm.compute((M), lambda i: A[i] * 3.14, name='B')
    s = tvm.create_schedule(B.op)
    
    # schedule

    return s, [A, B]

# @autotvm.template
# def matmul_autotuner(N, L, M, dtype):
#     A = tvm.placeholder((N, L), name='A', dtype=dtype)
#     B = tvm.placeholder((L, M), name='B', dtype=dtype)

#     k = tvm.reduce_axis((0, L), name='k')
#     C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
#     s = tvm.create_schedule(C.op)

#     # schedule
#     y, x = s[C].op.axis
#     k = s[C].op.reduce_axis[0]

#     ##### define space begin #####
#     cfg = autotvm.get_config()
#     cfg.define_split("tile_y", y, num_outputs=2)
#     cfg.define_split("tile_x", x, num_outputs=2)
#     ##### define space end #####

#     # schedule according to config
#     yo, yi = cfg["tile_y"].apply(s, C, y)
#     xo, xi = cfg["tile_x"].apply(s, C, x)

#     s[C].reorder(yo, xo, k, yi, xi)

#     return s, [A, B, C]