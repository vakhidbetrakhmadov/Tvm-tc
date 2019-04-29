import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser(
        description='TC arguments'
    )

    parser.add_argument(
        '--debug', type=lambda x: (str(x) == 'True'), default=False,
         help='Run in debug mode.',
    )

    parser.add_argument(
        '--prog', type=str, default='matmul',
         help='Program to run.',
    )

    parser.add_argument(
        '--autotuner', type=lambda x: (str(x) == 'True'), default=False,
        help='Use autotuner to find best mapping options',
    )
    
    parser.add_argument(
        '--load_from_cache', type=lambda x: (str(x) == 'True'), default=False,
        help='Load autotuned mapping options from cache.',
    )
    parser.add_argument(
        '--store_to_cache', type=lambda x: (str(x) == 'True'), default=False,
        help='Store autotuned mapping options to cache.',
    )    
    parser.add_argument(
        '--tuner_threads', type=int, default=16,
        help='Number of CPU tuning threads.',
    )
    parser.add_argument(
        '--tuner_generations', type=int, default=25,
        help='Number of tuning generations.',
    )
    parser.add_argument(
        '--tuner_pop_size', type=int, default=100,
        help='Number candidates per tuning generations.',
    )
    parser.add_argument(
        '--tuner_number_elites', type=int, default=5,
        help='Number of best tuning candidates that survive each generation.',
    )
    parser.add_argument(
        '--tuner_devices', type=str, default='0',
        help='Comma separated list of tuning devices.',
    )
    parser.add_argument(
        '--tuner_cache_file',
        type=str,
        default='/tmp/cache_condensenet',
        help='File to store tuned mapping options',
    )

    parser.add_argument(
        '--mapToBlocks', type=int, nargs='+',
        help='The configuration of CUDA grid, i.e. the number of CUDA blocks along three dimensions.\
              Must be within the range allowed by CUDA (maximum 2^31-1 for the first value and 65535 for the second and third).\
              Note that TC mapper eliminates empty blocks and the actual launch size may be smaller than requested.',
    )
    parser.add_argument(
        '--mapToThreads', type=int, nargs='+',
        help='The configuration of CUDA block, i.e. the number of CUDA threads in each block along three dimensions.\
              Must be within the range allowed by CUDA (maximum 1024 for the first and second value, 32 for the third, product below 1024).\
              Note that TC mapper eliminates empty threads and the actual launch size may be smaller than requested.',
    )
    parser.add_argument(
        '--tile', type=int, nargs='+',
        help='Perform loop tiling on the generated code with the given sizes. Independent of mapping to a grid of thread blocks.',
    )
    parser.add_argument(
        '--useSharedMemory', type=lambda x: (str(x) == 'True'),
        help='Create block-local copies of data in shared memory when this can leverage data reuse or global memory access coalescing.',
    )
    parser.add_argument(
        '--maxSharedMemory', type=int,
        help='The amount of shared memory to use, in bytes. If not provided, TC will query the active GPU and use all available shared memory.',
    )
    parser.add_argument(
        '--unroll', type=int,
        help='Perform loop unrolling on the generated code and produce at most the given number of statements.',
    )
    parser.add_argument(
        '--unrollCopyShared', type=lambda x: (str(x) == 'True'),
        help='Also unroll the copies to and from shared memory introduced by the TC mapper. If unroll value is not provided, has no effect.',
    )
    parser.add_argument(
        '--useReaOnlyCache', type=lambda x: (str(x) == 'True'),
        help='Emit loads to the readonly cache when appropriate.',
    )
    parser.add_argument(
        '--matchLibraryCalls', type=lambda x: (str(x) == 'True'),
        help='Replace computation patterns with calls to highly optimized libraries (such as CUB, CUTLASS) when possible.',
    )
    parser.add_argument(
        '--fixParametersBeforeScheduling', type=lambda x: (str(x) == 'True'),
        help='Perform automatic loop scheduling taking into account specific tensor sizes.\
              May produce faster kernels but significantly increases compilation time. Note that the mapping will be performed for specific tensor sizes anyway.',
    )
    parser.add_argument(
        '--outerScheduleFusionStrategy', type=str,
        help='Require TC to try and execute different TC expressions interleaved (Max),\
              separately (Min) or interleaved as long as sufficient parallelism is exploited (Preserve3Coincident)\
              by performing loop fusion and fission. Applies before tiling.',
    )
    parser.add_argument(
        '--intraTileFusionStrategy', type=str,
        help='Require TC to try and execute different TC expressions interleaved (Max),\
              separately (Min) or interleaved as long as sufficient parallelism is\
              exploited (Preserve3Coincident) by performing loop fusion and fission. Applies to inner loops created by tiling.',
    )

    return parser
