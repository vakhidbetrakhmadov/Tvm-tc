import time
import torch
import programs
import tensor_comprehensions as tc 

from args_parser import get_argument_parser
from builder import build

parser = get_argument_parser()
args, extra_args = parser.parse_known_args()

tc.dump_cuda(True)

torch.backends.cudnn.benchmark = True

if args.prog == 'matmul':
    M, K, N = 50, 50, 50

    A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()

    matmul = build(args, programs.MATMUL, args.prog, A, B)

    torch.cuda.synchronize()
    start = time.clock()
    C = matmul(A, B)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', C)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

elif args.prog == 'map':
    M = 50

    A = torch.randn(M).cuda()
    
    _map = build(args, programs.MAP, args.prog, A)

    torch.cuda.synchronize()
    start = time.clock()
    B = _map(A)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', B)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

elif args.prog == 'conv2d':
    batch = 256
    in_channel = 256
    out_channel = 512
    in_size = 14
    kernel = 3
    stride = 1
    padding = 0
    out_size = (in_size - kernel + 2 * padding) // stride + 1

    IN = torch.randn(batch, in_channel, in_size, in_size).cuda()
    WEIGHT = torch.randn(out_channel, in_channel, kernel, kernel).cuda()

    conv2d = build(args, programs.CONV2D, args.prog, IN, WEIGHT)

    torch.cuda.synchronize()
    start = time.clock()
    OUT = conv2d(IN, WEIGHT)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', OUT)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

elif args.prog == 'tmm':
    M, K, N = 50, 50, 50
    
    A, B = torch.randn(M, K).cuda(), torch.randn(N, K).cuda()

    conv = build(args, programs.TMM, args.prog, A, B)

    torch.cuda.synchronize()
    start = time.clock()
    C = tmm(A, B)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', C)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

elif args.prog == 'tbmm':
    B, M, K, N = 50, 50, 50, 50

    A, B = torch.randn(B, N, M).cuda(), torch.randn(B, K, M).cuda()

    tbmm = build(args, programs.TBMM, args.prog, A, B)

    torch.cuda.synchronize()
    start = time.clock()
    C = tbmm(A, B)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', C)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))
