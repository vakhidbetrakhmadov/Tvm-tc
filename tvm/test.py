import numpy as np
import tvm

target_host = "llvm"
target = "cuda"

N = 50

A = tvm.placeholder((N), name='A', dtype='float32')
B = tvm.compute((N), lambda i: A[i] * 3.14, name='B')
s = tvm.create_schedule(B.op)

x = s[B].op.axis
s[B].bind(x, tvm.thread_axis("threadIdx.x")

test = tvm.build(s, [A, B]], target, target_host=target_host, name="test")

ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(N)).astype(np.float32)
a_tvm = tvm.ndarray(a_np, ctx=ctx)
b_tvm = tvm.ndarray.empty((N), ctx=ctx)

test(a_tvm, b_tvm)

if target == "cuda" or target.startswith('opencl'):
    dev_module = test.imported_modules[0]
    print(dev_module.get_source())
else:
    print(test.get_source())