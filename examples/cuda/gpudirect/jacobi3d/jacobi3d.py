from charm4py import *
from numba import cuda
from argparse import ArgumentParser
from enum import Enum
import time

class Defaults(Enum):
    GRID_WIDTH = 512,
    GRID_HEIGHT = 512,
    GRID_DEPTH = 512,
    NUM_ITERS = 512,
    WARMUP_ITERS = 10,
    USE_ZEROCOPY = False
    PRINT_ELEMENTS = False


def main(args):
    Defaults.NUM_CHARES = charm.numPes()
    argp = ArgumentParser(description = "Jacobi3D implementation in Charm4Py using "
                          "CUDA and GPU-Direct communication"
                          )
    argp.add_argument('-x', '--grid_width', help = "Grid width",
                      default = Defaults.GRID_WIDTH.value
                      )
    argp.add_argument('-y', '--grid_height', help = "Grid height",
                       default = Defaults.GRID_HEIGHT.value
                       )
    argp.add_argument('-z', '--grid_depth', help = "Grid depth",
                      default = Defaults.GRID_DEPTH.value
                      )
    argp.add_argument('-c', '--num_chares', help = "Number of chares",
                       default = Defaults.NUM_CHARES
                       )
    argp.add_argument('-i', '--iterations', help = "Number of iterations",
                      default = Defaults.NUM_ITERS.value
                      )
    argp.add_argument('-w', '--warmup_iterations', help = "Number of warmup iterations",
                      default = Defaults.WARMUP_ITERS.value
                      )
    argp.add_argument('-d', '--use_zerocopy', action = "store_true",
                      help = "Use zerocopy when performing data transfers",
                      default = Defaults.USE_ZEROCOPY.value
                      )
    argp.add_argument('-p', '--print_blocks', help = "Print blocks",
                      action = "store_true",
                      default = Defaults.PRINT_ELEMENTS.value
                      )
    args = argp.parse_args()

    # charm.exit()

# charm.start(main)
if __name__ == '__main__':
    main(None)
