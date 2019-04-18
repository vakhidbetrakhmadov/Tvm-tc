import time
import torch
import tensor_comprehensions as tc 

from args_parser import get_argument_parser
from generate_options import generate_options

parser = get_argument_parser()
args, extra_args = parser.parse_known_args()

torch.backends.cudnn.benchmark = True

MATMUL = """
def matmul(float(M, K) A, float(K, N) B) -> (C) {
    C(m, n) +=! A(m, r_k) * B(r_k, n)
}
"""

TC = tc.define(MATMUL, generate_options(args))

M, K, N = 50, 50, 50

A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()

torch.cuda.synchronize()
start = time.clock()
C = TC.matmul(A, B)
torch.cuda.synchronize()
end = time.clock()

print('Result: ', C)
print('Execution time: ', start - end)