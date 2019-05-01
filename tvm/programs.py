import tvm
import argparse

from tvm import autotvm

def matmul(args):
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
    x = s[B].op.axis[0]

    s[B].bind(x, tvm.thread_axis("threadIdx.x"))

    return s, [A, B]

def conv2d(in_size, in_channel, batch, kernel, out_channel, stride, args):
    pad = 0
    out_size = (in_size - kernel + 2*pad) // stride + 1

    A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
    
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = tvm.compute((out_size, out_size, out_channel, batch),
                    lambda yy, xx, ff, nn: tvm.sum(A[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
                                                   axis=[ry, rx, rc]),
                    name='B')

    # Designate the memory hierarchy
    s = tvm.create_schedule(B.op)
    
    AA = s.cache_read(A, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    # tile consts
    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    # Get the GPU thread indices
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    by, fi = s[B].split(fi, factor=block_factor)
    bx, ni = s[B].split(ni, factor=block_factor)

    # Bind the iteration variables to GPU thread indices
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)

    tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, ni = s[B].split(ni, nparts=num_thread)
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    yi, xi, fi, ni = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    s[BL].reorder(rco, ry, rx, rci, fi, ni)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    yi, xi, ci, fi = s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    _, fi = s[WW].split(fi, factor=4)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    s[WW].vectorize(fi)  # vectorize memory load

    return s, [A, W, B]

def tmm(args):
    M, K, N  = tvm.var("M"), tvm.var("K"), tvm.var("N")
    A = tvm.placeholder((M, K), name='A', dtype="float32")
    B = tvm.placeholder((N, K), name='B', dtype="float32")

    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    x, y = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    xo, xi = s[C].split(x, args.x)
    yo, yi = s[C].split(y, args.y)
    
    # s[C].reorder(xo, yo, k, xi, yi) ... 

    s[C].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[C].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[C].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[C].bind(yi, tvm.thread_axis("threadIdx.y"))

    return s, [A, B, C]

def tbmm(args):
    B, N, M, K  = tvm.var("B"), tvm.var("N"), tvm.var("M"), tvm.var("K")

    X = tvm.placeholder((B, N, M), name='X', dtype="float32")
    Y = tvm.placeholder((B, K, M), name='Y', dtype="float32")

    m = tvm.reduce_axis((0, M), name='m')
    Z = tvm.compute((B, N, K), lambda b, n, k: tvm.sum(X[b, n, m] * Y[b, k, m], axis=m), name='Z')
    s = tvm.create_schedule(Z.op)

    x, y, z = s[Z].op.axis
    m = s[Z].op.reduce_axis[2]

    xo, xi = s[Z].split(x, args.x)
    yo, yi = s[Z].split(y, args.y)
    zo, zi = s[Z].split(z, args.z)

    # s[Z].reorder(xo, yo, k, xi, yi) ... 

    s[Z].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[Z].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[Z].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[Z].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[Z].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[Z].bind(zi, tvm.thread_axis("threadIdx.z"))

    return s, [X, Y, Z]










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