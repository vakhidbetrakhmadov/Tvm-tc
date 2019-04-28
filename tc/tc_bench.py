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

elif args.prog == 'conv':
    batch = 256
    in_channel = 256
    out_channel = 512
    in_size = 14
    kernel = 3
    stride = 1
    padding = 0
    out_size = (in_size - kernel + 2 * padding) // stride + 1

    TC = tc.define(programs.CONV, generate_options(args))

    IN = torch.randn(batch, in_channel, in_size, in_size).cuda()
    WEIGHT = torch.randn().cuda(out_channel, in_channel, kernel, kernel)

    torch.cuda.synchronize()
    start = time.clock()
    OUT = TC.map(IN, WEIGHT)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', OUT)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))