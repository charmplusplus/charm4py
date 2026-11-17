"""3-D Jacobi with the ghost exchange on the GPU (numba.cuda kernels), one block per
chare. Ported from charm4py PR #208 to the Charm4py 2 API; see README.md.

Run (one node, two GPUs):
    python -m charmrun.start +p2 jacobi3d.py -x 256 -y 256 -z 256 -m device
    srun --mpi=pmix -n 2 --gpus=2 --gpu-bind=none env CUDA_VISIBLE_DEVICES=0,1 \\
        python jacobi3d.py -x 256 -y 256 -z 256 -m persistent +pe 2
Check without a GPU (parses the options, prints the decomposition, exits):
    python -m charmrun.start +p2 jacobi3d.py --dry-run
"""
from charm4py import charm, Array, Future
from argparse import ArgumentParser
from functools import reduce
import numpy as np
import sys

from block import Block

MODES = ('staged', 'device', 'persistent')


def parse(argv, num_pes):
    argp = ArgumentParser(description='Jacobi3D in Charm4py with CUDA kernels (numba) and '
                          'device-to-device ghost exchange')
    argp.add_argument('-x', '--grid_width', help='grid width', type=int, default=512)
    argp.add_argument('-y', '--grid_height', help='grid height', type=int, default=512)
    argp.add_argument('-z', '--grid_depth', help='grid depth', type=int, default=512)
    argp.add_argument('-c', '--num_chares', help='number of chares (default: number of PEs)',
                      type=int, default=num_pes)
    argp.add_argument('-i', '--iterations', help='number of timed iterations', type=int, default=100)
    argp.add_argument('-w', '--warmup_iterations', help='number of warm-up iterations', type=int, default=10)
    argp.add_argument('-m', '--mode', choices=MODES, default='device',
                      help='ghost exchange: staged (through pinned host memory), device '
                      '(device arrays as arguments into posted receives), persistent '
                      '(charm.DevicePersistent put plus a notification message); default device')
    argp.add_argument('-d', '--use_zerocopy', action='store_true',
                      help='PR #208 option for device-to-device transfers: same as -m device')
    argp.add_argument('--validate', action='store_true',
                      help='compare the sum of the final grid with a numpy computation on PE 0 '
                      '(small grids only)')
    argp.add_argument('--dry-run', action='store_true',
                      help='print the configuration and exit without touching a GPU')
    # Charm++ options (+pe, +p, ++ppn) are left to Charm++
    args, _ = argp.parse_known_args([a for a in argv[1:] if not a.startswith('+')])
    if args.use_zerocopy:
        args.mode = 'device'
    return args


def main(argv):
    args = parse(argv, charm.numPes())
    w, h, d = args.grid_width, args.grid_height, args.grid_depth
    num_chares = args.num_chares
    dims = calc_num_chares_per_dim(num_chares, w, h, d)
    nx, ny, nz = dims
    if reduce(lambda a, b: a * b, dims) != num_chares or w % nx or h % ny or d % nz:
        print(f'ERROR: bad grid of chares: {nx} x {ny} x {nz} for {num_chares} chares and a '
              f'{w} x {h} x {d} grid (the grid must divide evenly)')
        charm.exit(1)
        return
    bw, bh, bd = w // nx, h // ny, d // nz

    print('\n[CUDA 3D Jacobi example]\n')
    print(f'Grid: {w} x {h} x {d}, Block: {bw} x {bh} x {bd}, '
          f'Chares: {nx} x {ny} x {nz}, Iterations: {args.iterations}, '
          f'Warm-up: {args.warmup_iterations}, Mode: {args.mode}, PEs: {charm.numPes()}\n')
    if args.dry_run:
        charm.exit()
        return
    try:
        import numba.cuda   # noqa: F401
    except ImportError:
        print('jacobi3d: numba (with numba-cuda) is not installed')
        charm.exit(1)
        return
    from charm4py import _ckdyn
    if args.mode != 'staged' and not _ckdyn.has_device_support():
        print(f'jacobi3d: -m {args.mode} needs a CUDA build of Charm++; -m staged does not')
        charm.exit(1)
        return

    cfg = {'mode': args.mode, 'n_chares': (nx, ny, nz),
           'block_width': bw, 'block_height': bh, 'block_depth': bd,
           'n_iters': args.iterations, 'warmup_iters': args.warmup_iterations}
    init_done = Future()
    blocks = Array(Block, dims=[nx, ny, nz], args=[cfg, init_done])
    init_done.get()
    done = Future()
    blocks.run(done)
    total = done.get()
    print(f'Sum of the interior after {args.warmup_iterations + args.iterations} iterations: {total:.10e}')
    if args.validate:
        ref = reference_sum(w, h, d, args.warmup_iterations + args.iterations)
        ok = abs(total - ref) <= 1e-9 * max(abs(ref), 1.0)
        print(f'Validation: numpy sum {ref:.10e}, {"PASS" if ok else "FAIL"}')
        if not ok:
            charm.exit(1)
            return
    charm.exit()


def reference_sum(w, h, d, iters):
    """ the same computation on the whole grid with numpy (index order z, y, x) """
    t = np.zeros((d + 2, h + 2, w + 2))
    t[1:-1, 1:-1, 0] = t[1:-1, 1:-1, -1] = 1.0
    t[1:-1, 0, 1:-1] = t[1:-1, -1, 1:-1] = 1.0
    t[0, 1:-1, 1:-1] = t[-1, 1:-1, 1:-1] = 1.0
    new = t.copy()
    c = (slice(1, -1),) * 3
    for _ in range(iters):
        new[c] = (t[c] + t[1:-1, 1:-1, :-2] + t[1:-1, 1:-1, 2:] + t[1:-1, :-2, 1:-1] +
                  t[1:-1, 2:, 1:-1] + t[:-2, 1:-1, 1:-1] + t[2:, 1:-1, 1:-1]) * 0.142857
        t, new = new, t
    return float(t[c].sum())


def calc_num_chares_per_dim(num_chares, grid_w, grid_h, grid_d):
    """ the decomposition with the smallest surface (from PR #208) """
    n_chares = [0, 0, 0]
    area = [grid_w * grid_h, grid_w * grid_d, grid_h * grid_d]
    bestsurf = 2.0 * sum(area)
    ipx = 1
    while ipx <= num_chares:
        if not num_chares % ipx:
            nremain = num_chares // ipx
            ipy = 1
            while ipy <= nremain:
                if not nremain % ipy:
                    ipz = nremain // ipy
                    surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz
                    if surf < bestsurf:
                        bestsurf = surf
                        n_chares = [ipx, ipy, ipz]
                ipy += 1
        ipx += 1
    return n_chares


charm.start(main, modules=['block'])
