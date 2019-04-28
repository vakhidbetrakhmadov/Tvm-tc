import time
import torch
import programs
import tensor_comprehensions as tc 

from args_parser import get_argument_parser
from generate_options import generate_options

parser = get_argument_parser()
args, extra_args = parser.parse_known_args()

tc.dump_cuda(True)

torch.backends.cudnn.benchmark = True

if args.prog == 'matmul':
    M, K, N = 50, 50, 50

    TC = tc.define(programs.MATMUL, generate_options(args))

    A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()

    torch.cuda.synchronize()
    start = time.clock()
    C = TC.matmul(A, B)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', C)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

elif args.prog == 'map':
    M = 50

    TC = tc.define(programs.MAP, generate_options(args))

    A = torch.randn(M).cuda()

    torch.cuda.synchronize()
    start = time.clock()
    B = TC.map(A)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', B)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

elif args.prog == 'reduce':
    M = 50

    TC = tc.define(programs.REDUCE, generate_options(args))

    A = torch.randn(M).cuda()

    torch.cuda.synchronize()
    start = time.clock()
    B = TC.reduce(A)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', B)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))