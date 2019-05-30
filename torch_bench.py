# import tensorflow as tf
# import numpy as np
import torch
import time

torch.backends.cudnn.benchmark = True

count = 10

def run_and_time(operation):
    total_time = 0
    for _ in range(0, count):
        torch.cuda.synchronize()
        start = time.clock()
        operation()
        torch.cuda.synchronize()
        end = time.clock()
        total_time += (end - start)

    print('Execution time: {} ms'.format((total_time / count) * 10 ** 3))

def main():
    print('Started')

    def test_matmul():
        sizes = [(72, 26, 26),
                (50, 50, 50),
                (128, 32, 256),
                (128, 1024, 1024), 
                (128, 4096, 16384)]
                
        for size in sizes:
            M, K, N = size
            A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()
            
            def matmul():
                o = torch.mm(A, B)

            print('Program: matmul. Size: {}'.format(size))

            run_and_time(matmul)

    def test_tmm():
        sizes = [(72, 26, 26),
                (50, 50, 50),
                (128, 32, 256),
                (128, 1024, 1024), 
                (128, 4096, 16384)]

        for size in sizes:
            M, K, N = size
            A, B = torch.randn(M, K).cuda(), torch.randn(N, K).cuda()

            def tmm():
                o = torch.mm(A, B.t())

            print('Program: tmm. Size: {}'.format(size))

            run_and_time(tmm)

    def test_tbmm():
        sizes = [(72, 26, 26, 250), 
                (72, 26, 26, 500), 
                (72, 26, 26, 1024), 
                (50, 50, 50, 500), 
                (50, 50, 50, 1024)]

        for size in sizes:
            M, K, N, B = size
            A, B = torch.randn(B, N, M).cuda(), torch.randn(B, K, M).cuda()

            def tbmm():
                o = torch.bmm(A, B.transpose(1, 2))

            print('Program: tbmm. Size: {}'.format(size))

            run_and_time(tbmm)

    def test_conv2d():
        sizes = [(32, 16, 16, 14, 14),
                (32, 32, 32, 7, 7),
                (32, 4, 4, 56, 56),
                (32, 8, 8, 28, 28),
                (256, 256, 512, 14, 3)]

        for size in sizes:
            batch, in_channel, out_channel, in_size, kernel = size
            stride = 1
            padding = 0

            IN = torch.randn(batch, in_channel, in_size, in_size).cuda()
            WEIGHT = torch.randn(in_channel, out_channel, kernel, kernel).cuda()

            def conv2d():
                torch.nn.functional.conv_transpose2d(IN, WEIGHT, stride=stride, padding=padding)
            
            print('Program: conv2d. Size: {}'.format(size))

            run_and_time(conv2d)

    def test_map():
        def apply(func, M):
            tList = [func(m) for m in torch.unbind(M, dim=0)]
            res = torch.stack(tList, dim=0)
            return res 

        sizes = [(1000),
                (10000),
                (100000),
                (1000000),
                (10000000)]
        
        for size in sizes:
            M = size
            A = torch.randn(M).cuda()

            def _map():
                # o = tf.map_fn(lambda x: x * 3.14, A)
                apply(lambda x: x * 3.14, A)

            print('Program: map. Size: {}'.format(size))

            run_and_time(_map)

    test_matmul()
    test_tmm()
    test_tbmm()
    test_conv2d()
    test_map()

    print('Ended')
    
main()