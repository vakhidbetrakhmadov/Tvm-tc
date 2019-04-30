import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser(
        description='TVM arguments'
    )

    parser.add_argument(
        '--debug', type=lambda x: (str(x) == 'False'), default=False,
         help='Run in debug mode.',
    )

    parser.add_argument(
        '--prog', type=str, default='matmul',
         help='Program to run.',
    )

    parser.add_argument(
        '--autotuner', type=lambda x: (str(x) == 'False'), default=False,
        help='Use autotuner to find best parameters',
    )

    parser.add_argument(
        '-x', type=int, default=8,
        help='Tiling factor for x axis for split schedule primitive.',
    )
    parser.add_argument(
        '-y', type=int, default=8,
        help='Tiling factor for y axis for split schedule primitive.',
    )
    parser.add_argument(
        '-z', type=int, default=8,
        help='Tiling factor for z axis for split schedule primitive.',
    )

    return parser
