"""One block of the 3-D Jacobi grid per chare, with its ghost exchange on the GPU.

Ported from charm4py PR #208 (Charm4py 1.x, Cython layer) to the Charm4py 2 API.
Three exchange modes (chosen in jacobi3d.py with -m):

staged      pack on the GPU, copy the ghost to pinned host memory, send the numpy
            array, copy it to the device on the receiver (PR #208 without -d).
device      send the device ghost array as an argument (a DEVICE slot, device to
            device); the receiver has posted a device array for that argument with
            Chare.post_buffer, so the message lands in it (PR #208 with -d, which
            used gpu_src_ptrs and post_buf_addresses on channels).
persistent  each block registers its receive ghosts once with charm.DevicePersistent
            and sends the handles to its neighbors; a neighbor puts its packed face
            straight into that buffer and then sends a small message with the
            iteration number (the counterpart of the persistent GPU-direct buffers
            of the Charm++ Jacobi3D).

The ghost of direction d is the face of the block on side d. A ghost a block
receives is unpacked into the halo on the side it came from, so the face a block
packs for its LEFT neighbor is received by that neighbor as its RIGHT ghost.

Ordering. A neighbor can be one iteration ahead (it has all the ghosts of the
current iteration and may send those of the next), never two. A ghost of the next
iteration is kept in self.early and unpacked when this block reaches that
iteration. Buffers that a transfer may still be reading or writing are doubled by
iteration parity:
- staged and device modes: the send ghosts. A send is complete when the neighbor
  has the message; when this block packs iteration i+2 into the same buffer it
  has received the neighbor's ghost of iteration i+1, which the neighbor sends
  only after it received the ghost of iteration i.
- persistent mode: the receive ghosts. A neighbor puts iteration i+2 into the
  buffer of iteration i only after it received this block's ghost of iteration
  i+1, which this block sends after it unpacked iteration i.
In device mode the posted receive buffer is single: it is posted again only after
its contents were unpacked (and the stream synchronized), and a neighbor cannot
send the next ghost of that direction before it has this block's ghost of the
next iteration, which this block sends after posting again.
"""
from charm4py import charm, Chare, Future, Reducer, coro
import numpy as np
import time

try:
    from numba import cuda
    import kernels
except ImportError:   # jacobi3d.py checks for numba before creating any block
    cuda = None
    kernels = None

LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK, DIR_COUNT = 0, 1, 2, 3, 4, 5, 6
OPPOSITE = (RIGHT, LEFT, BOTTOM, TOP, BACK, FRONT)
DIR_NAMES = ('left', 'right', 'top', 'bottom', 'front', 'back')


class Block(Chare):

    def __init__(self, cfg, init_done):
        self.cfg = cfg
        self.mode = cfg['mode']
        self.bw, self.bh, self.bd = cfg['block_width'], cfg['block_height'], cfg['block_depth']
        x, y, z = self.thisIndex
        nx, ny, nz = cfg['n_chares']
        # neighbor index per direction, None on the global boundary
        self.nbr = [(x-1, y, z) if x > 0 else None, (x+1, y, z) if x < nx-1 else None,
                    (x, y-1, z) if y > 0 else None, (x, y+1, z) if y < ny-1 else None,
                    (x, y, z-1) if z > 0 else None, (x, y, z+1) if z < nz-1 else None]
        self.bounds = [n is None for n in self.nbr]
        self.active = [d for d in range(DIR_COUNT) if not self.bounds[d]]
        self.neighbors = len(self.active)
        bw, bh, bd = self.bw, self.bh, self.bd
        self.ghost_counts = (bh*bd, bh*bd, bw*bd, bw*bd, bw*bh, bw*bh)

        self.iter = -1         # the iteration whose ghosts are being received
        self.count = 0         # ghosts of self.iter unpacked so far
        self.early = {}        # direction -> ghost of iteration self.iter+1
        self.wait_fut = None
        self.peer_handles = {}  # persistent mode: direction -> neighbor's receive handles
        self.handles_fut = None

        dev = charm.myDevice()
        try:
            cur = cuda.get_current_device().id
        except AttributeError:   # the numba CUDA simulator (NUMBA_ENABLE_CUDASIM=1)
            cur = dev
        if dev >= 0 and cur != dev:
            print('Block %s on PE %d: numba current device %d, charm.myDevice() %d'
                  % (self.thisIndex, charm.myPe(), cur, dev))
        self.stream = cuda.stream()
        n = (bw+2) * (bh+2) * (bd+2)
        self.d_temperature = cuda.device_array(n, dtype=np.float64, stream=self.stream)
        self.d_new_temperature = cuda.device_array(n, dtype=np.float64, stream=self.stream)
        for t in (self.d_temperature, self.d_new_temperature):
            kernels.invokeInitKernel(t, bw, bh, bd, self.stream)
            kernels.invokeBoundaryKernels(t, bw, bh, bd, self.bounds, self.stream)

        def dev_ghosts():
            return [cuda.device_array(self.ghost_counts[d], dtype=np.float64, stream=self.stream)
                    if not self.bounds[d] else None for d in range(DIR_COUNT)]
        if self.mode == 'persistent':
            self.d_send = [dev_ghosts()]
            self.d_recv = [dev_ghosts(), dev_ghosts()]
        else:
            self.d_send = [dev_ghosts(), dev_ghosts()]
            self.d_recv = [dev_ghosts()]
        if self.mode == 'staged':
            self.h_send = [[cuda.pinned_array(self.ghost_counts[d], dtype=np.float64)
                            if not self.bounds[d] else None for d in range(DIR_COUNT)] for _ in range(2)]
        self.stream.synchronize()

        if self.mode == 'device':
            for d in self.active:
                self.post_buffer(self.recv_method(d), 1, self.d_recv[0][d])
        elif self.mode == 'persistent':
            self.p_send = [charm.DevicePersistent(self.d_send[0][d]) if not self.bounds[d] else None
                           for d in range(DIR_COUNT)]
            self.p_recv = [[charm.DevicePersistent(self.d_recv[par][d]) if not self.bounds[d] else None
                            for d in range(DIR_COUNT)] for par in range(2)]
            for d in self.active:
                # the neighbor in direction d receives these as the ghosts from OPPOSITE[d]
                self.thisProxy[self.nbr[d]].set_handles(OPPOSITE[d], [self.p_recv[0][d].handle,
                                                                      self.p_recv[1][d].handle])
        self.reduce(init_done)

    # one entry method per direction: a post covers one (method, argument) pair
    def recv_method(self, d):
        return (self.recv_left, self.recv_right, self.recv_top,
                self.recv_bottom, self.recv_front, self.recv_back)[d]

    def recv_left(self, it, ghost):
        self.recv_ghost(LEFT, it, ghost)

    def recv_right(self, it, ghost):
        self.recv_ghost(RIGHT, it, ghost)

    def recv_top(self, it, ghost):
        self.recv_ghost(TOP, it, ghost)

    def recv_bottom(self, it, ghost):
        self.recv_ghost(BOTTOM, it, ghost)

    def recv_front(self, it, ghost):
        self.recv_ghost(FRONT, it, ghost)

    def recv_back(self, it, ghost):
        self.recv_ghost(BACK, it, ghost)

    def ghost_ready(self, d, it):
        """ persistent mode: the neighbor on side d has put its ghost of iteration it """
        self.recv_ghost(d, it, None)

    def set_handles(self, d, handles):
        self.peer_handles[d] = handles
        if len(self.peer_handles) == self.neighbors and self.handles_fut is not None:
            self.handles_fut.send(None)

    def recv_ghost(self, d, it, ghost):
        if it == self.iter:
            self.unpack(d, it, ghost)
            self.count += 1
            if self.count == self.neighbors and self.wait_fut is not None:
                self.wait_fut.send(None)
        else:
            assert it == self.iter + 1, 'block %s: ghost of iteration %d from %s at iteration %d' \
                % (self.thisIndex, it, DIR_NAMES[d], self.iter)
            self.early[d] = ghost

    def unpack(self, d, it, ghost):
        if self.mode == 'persistent':
            g = self.d_recv[it % 2][d]
        elif self.mode == 'staged':
            g = self.d_recv[0][d]
            g.copy_to_device(ghost, stream=self.stream)
        else:
            posted = ghost is self.d_recv[0][d]
            g = ghost if posted else cuda.as_cuda_array(ghost)   # a DeviceBuffer without a post
        kernels.invokeUnpackingKernel(self.d_temperature, g, d, self.bw, self.bh, self.bd, self.stream)
        if self.mode == 'device':
            # the next message of this direction may land in the posted buffer: read it first
            self.stream.synchronize()
            self.post_buffer(self.recv_method(d), 1, self.d_recv[0][d])

    def send_ghosts(self, it):
        """ called from run (a coroutine): DevicePersistent.put returns a Future """
        par = it % 2
        for d in self.active:
            src = self.d_send[0 if self.mode == 'persistent' else par][d]
            kernels.invokePackingKernel(self.d_temperature, src, d, self.bw, self.bh, self.bd, self.stream)
            if self.mode == 'staged':
                src.copy_to_host(self.h_send[par][d], stream=self.stream)
        self.stream.synchronize()
        if self.mode == 'persistent':
            futs = [(d, self.p_send[d].put(self.peer_handles[d][par])) for d in self.active]
            for d, f in futs:
                f.get()
                self.thisProxy[self.nbr[d]].ghost_ready(OPPOSITE[d], it)
        else:
            for d in self.active:
                buf = self.h_send[par][d] if self.mode == 'staged' else self.d_send[par][d]
                getattr(self.thisProxy[self.nbr[d]], 'recv_' + DIR_NAMES[OPPOSITE[d]])(it, buf)

    @coro
    def run(self, done_future):
        cfg = self.cfg
        n_iters, warmup = cfg['n_iters'], cfg['warmup_iters']
        if self.mode == 'persistent':
            if len(self.peer_handles) < self.neighbors:
                self.handles_fut = Future()
                self.handles_fut.get()
            if self.thisIndex == (0, 0, 0) and self.active:
                d = self.active[0]
                print('Persistent transfer mode of block (0, 0, 0) to its %s neighbor: %s'
                      % (DIR_NAMES[d], self.p_send[d].mode(self.peer_handles[d][0])))
        comm_time = 0.0
        tstart = time.perf_counter()
        for it in range(n_iters + warmup):
            if it == warmup:
                tstart = time.perf_counter()
                comm_time = 0.0
            t0 = time.perf_counter()
            self.iter = it
            self.count = 0
            for d in list(self.early):
                self.unpack(d, it, self.early.pop(d))
                self.count += 1
            self.send_ghosts(it)
            if self.count < self.neighbors:
                self.wait_fut = Future()
                self.wait_fut.get()
                self.wait_fut = None
            comm_time += time.perf_counter() - t0
            kernels.invokeJacobiKernel(self.d_temperature, self.d_new_temperature,
                                       self.bw, self.bh, self.bd, self.stream)
            self.stream.synchronize()
            self.d_temperature, self.d_new_temperature = self.d_new_temperature, self.d_temperature
        elapsed = time.perf_counter() - tstart

        if self.mode == 'persistent':
            for p in self.p_send + self.p_recv[0] + self.p_recv[1]:
                if p is not None:
                    p.close()
        bw, bh, bd = self.bw, self.bh, self.bd
        t = self.d_temperature.copy_to_host(stream=self.stream)
        self.stream.synchronize()
        interior = float(t.reshape(bd+2, bh+2, bw+2)[1:-1, 1:-1, 1:-1].sum())
        if self.thisIndex == (0, 0, 0):
            n = max(n_iters, 1)
            print(f'Elapsed time: {round(elapsed, 3)} s')
            print(f'Average time per iteration: {round(elapsed / n * 1e3, 3)} ms')
            print(f'Communication time per iteration: {round(comm_time / n * 1e3, 3)} ms')
        self.reduce(done_future, interior, Reducer.sum)
