import time
import torch
import programs
import tensor_comprehensions as tc 

from args_parser import get_argument_parser
from build import build_and_time

parser = get_argument_parser()
args, extra_args = parser.parse_known_args()

tc.dump_cuda(True)

torch.backends.cudnn.benchmark = True

def run_and_time(exe: tc.Executor, *inputs: torch.Tensor):
    torch.cuda.synchronize()
    start = time.clock()
    out = exe(*inputs)
    torch.cuda.synchronize()
    end = time.clock()

    print('Result: ', out)
    print('Execution time: {} ms'.format((end - start) * 10 ** 3))

def main():
    if args.prog == 'matmul':
        M, K, N = 50, 50, 50

        A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()

        matmul = build_and_time(args, programs.MATMUL, args.prog, A, B)
        run_and_time(matmul, A, B)
        
    elif args.prog == 'map':
        M = 50

        A = torch.randn(M).cuda()
        
        _map = build_and_time(args, programs.MAP, args.prog, A)
        run_and_time(matmul, _map)

    elif args.prog == 'conv2d':
        batch = 256
        in_channel = 256
        out_channel = 512
        in_size = 14
        kernel = 3
        stride = 1
        padding = 0

        IN = torch.randn(batch, in_channel, in_size, in_size).cuda()
        WEIGHT = torch.randn(out_channel, in_channel, kernel, kernel).cuda()

        conv2d = build_and_time(args, programs.CONV2D, args.prog, IN, WEIGHT)
        run_and_time(conv2d, IN, WEIGHT)

    elif args.prog == 'tmm':
        M, K, N = 50, 50, 50
        
        A, B = torch.randn(M, K).cuda(), torch.randn(N, K).cuda()

        tmm = build_and_time(args, programs.TMM, args.prog, A, B)
        run_and_time(tmm, A, B)

    elif args.prog == 'tbmm':
        B, M, K, N = 50, 50, 50, 50

        A, B = torch.randn(B, N, M).cuda(), torch.randn(B, K, M).cuda()

        tbmm = build_and_time(args, programs.TBMM, args.prog, A, B)
        run_and_time(tbmm, A, B)

main()