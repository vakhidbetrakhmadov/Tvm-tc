import numpy as np
import tvm

target_host = 'llvm'
target = 'cuda'

N = 50

A = tvm.placeholder((tvm.var('N')), name='A', dtype='float32')
B = tvm.compute((N), lambda i: A[i] * 3.14, name='B')
s = tvm.create_schedule(B.op)

x = s[B].op.axis
s[B].bind(x, tvm.thread_axis("threadIdx.x"))

t = tvm.build(s, [A, B], target, target_host=target_host, name="t")

ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(N)).astype(np.float32)
a_tvm = tvm.ndarray(a_np, ctx=ctx)
b_tvm = tvm.ndarray.empty((N), ctx=ctx)

t(a_tvm, b_tvm)

if target == "cuda" or target.startswith('opencl'):
    dev_module = t.imported_modules[0]
    print(dev_module.get_source())
else:
    print(t.get_source())