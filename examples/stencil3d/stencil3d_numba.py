# Port of examples/charm++/load_balancing/stencil3d from Charm++ codebase
# This version uses NumPy and Numba

from charm4py import charm, Chare, Group, Array, when
from charm4py import readonlies as ro
import time
import math
import numpy as np
import numba

import sys
sys.argv += ['+LBPeriod', '0.001', '+LBOff', '+LBCommOff']

from charm4py import Options
Options.PROFILING = False


MAX_ITER = 100
LBPERIOD_ITER = 30     # LB is called every LBPERIOD_ITER number of program iterations
CHANGELOAD = 30
LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK = range(6)
DIVIDEBY7 = 0.14285714285714285714
PRINT_ITERATIONS = False # print msg after each iteration


def main(args):

    if (len(args) != 3) and (len(args) != 7):
        print(args[0] + " [array_size] [block_size]")
        print("OR " + args[0] + " [array_size_X] [array_size_Y] [array_size_Z] [block_size_X] [block_size_Y] [block_size_Z]")
        charm.abort("Incorrect program arguments")

    if len(args) == 3:
        ro.arrayDimX = ro.arrayDimY = ro.arrayDimZ = int(args[1])
        ro.blockDimX = ro.blockDimY = ro.blockDimZ = int(args[2])
    elif len(args) == 7:
        ro.arrayDimX, ro.arrayDimY, ro.arrayDimZ = [int(arg) for arg in args[1:4]]
        ro.blockDimX, ro.blockDimY, ro.blockDimZ = [int(arg) for arg in args[4:7]]

    if (ro.arrayDimX < ro.blockDimX) or (ro.arrayDimX % ro.blockDimX != 0): charm.abort("array_size_X % block_size_X != 0!")
    if (ro.arrayDimY < ro.blockDimY) or (ro.arrayDimY % ro.blockDimY != 0): charm.abort("array_size_Y % block_size_Y != 0!")
    if (ro.arrayDimZ < ro.blockDimZ) or (ro.arrayDimZ % ro.blockDimZ != 0): charm.abort("array_size_Z % block_size_Z != 0!")

    ro.num_chare_x = ro.arrayDimX // ro.blockDimX
    ro.num_chare_y = ro.arrayDimY // ro.blockDimY
    ro.num_chare_z = ro.arrayDimZ // ro.blockDimZ

    print("\nSTENCIL COMPUTATION WITH BARRIERS\n")
    print("Running Stencil on " + str(charm.numPes()) + " processors with " + str((ro.num_chare_x, ro.num_chare_y, ro.num_chare_z)) + " chares")
    print("Array Dimensions: " + str((ro.arrayDimX, ro.arrayDimY, ro.arrayDimZ)))
    print("Block Dimensions: " + str((ro.blockDimX, ro.blockDimY, ro.blockDimZ)))

    nb_precomps = Group(NumbaPrecompiler)
    charm.awaitCreation(nb_precomps) # wait until Numba functions are compiled

    sim_done = charm.createFuture()
    array = Array(Stencil, (ro.num_chare_x, ro.num_chare_y, ro.num_chare_z), args=[sim_done])
    charm.awaitCreation(array)

    print("Starting simulation")
    initTime = time.time()
    array.start()
    sim_done.get() # wait until simulation completes
    totalTime = time.time() - initTime
    print(MAX_ITER, "iterations completed, total time=", round(totalTime,3), "secs, time per iteration (ms) =", round(totalTime / MAX_ITER * 1000, 3))
    charm.printStats()
    exit()


def make_numba_functions():

    if 'index' in globals(): return # numba functions already generated

    global blockDimX, blockDimY, blockDimZ
    # numba functions will be compiled with blockDimX, blockDimY, blockDimZ as constants
    blockDimX = ro.blockDimX
    blockDimY = ro.blockDimY
    blockDimZ = ro.blockDimZ

    @numba.jit(nopython=True, cache=False)
    def index(a,b,c): return (a + b*(blockDimX+2) + c*(blockDimX+2)*(blockDimY+2))

    @numba.jit(nopython=True, cache=False)
    def compute_kernel_fast(work, new_temperature, temperature):
        for k in range(1, blockDimZ+1):
            for j in range(1, blockDimY+1):
                for i in range(1, blockDimX+1):
                    for w in range(work):
                        # update my value based on the surrounding values
                        new_temperature[index(i, j, k)] = (temperature[index(i-1, j, k)] \
                            +  temperature[index(i+1, j, k)] \
                            +  temperature[index(i, j-1, k)] \
                            +  temperature[index(i, j+1, k)] \
                            +  temperature[index(i, j, k-1)] \
                            +  temperature[index(i, j, k+1)] \
                            +  temperature[index(i, j, k)] ) \
                            *  DIVIDEBY7

    @numba.jit(nopython=True, cache=False)
    def constrainBC_fast(T):
        # Heat left, top and front faces of each chare's block
        for k in range(1, blockDimZ+1):
            for i in range(1, blockDimX+1):
                T[index(i, 1, k)] = 255.0

        for k in range(1, blockDimZ+1):
            for j in range(1, blockDimY+1):
                T[index(1, j, k)] = 255.0

        for j in range(1, blockDimY+1):
            for i in range(1, blockDimX+1):
                T[index(i, j, 1)] = 255.0

    @numba.jit(nopython=True, cache=False)
    def fillGhostData(T, leftGhost, rightGhost, topGhost, bottomGhost, frontGhost, backGhost):
        for k in range(blockDimZ):
            for j in range(blockDimY):
                leftGhost[k*blockDimY+j] = T[index(1, j+1, k+1)]
                rightGhost[k*blockDimY+j] = T[index(blockDimX, j+1, k+1)]

        for k in range(blockDimZ):
            for i in range(blockDimX):
                topGhost[k*blockDimX+i] = T[index(i+1, 1, k+1)]
                bottomGhost[k*blockDimX+i] = T[index(i+1, blockDimY, k+1)]

        for j in range(blockDimY):
            for i in range(blockDimX):
                frontGhost[j*blockDimX+i] = T[index(i+1, j+1, 1)];
                backGhost[j*blockDimX+i] = T[index(i+1, j+1, blockDimZ)]

    @numba.jit(nopython=True, cache=False)
    def processGhosts_fast(T, direction, width, height, gh):
        if direction == LEFT:
            for k in range(width):
                for j in range(height):
                    T[index(0, j+1, k+1)] = gh[k*height+j]
        elif direction == RIGHT:
            for k in range(width):
                for j in range(height):
                    T[index(blockDimX+1, j+1, k+1)] = gh[k*height+j]
        elif direction == BOTTOM:
            for k in range(width):
                for i in range(height):
                    T[index(i+1, 0, k+1)] = gh[k*height+i]
        elif direction == TOP:
            for k in range(width):
                for i in range(height):
                    T[index(i+1, blockDimY+1, k+1)] = gh[k*height+i]
        elif direction == FRONT:
            for j in range(width):
                for i in range(height):
                    T[index(i+1, j+1, 0)] = gh[j*height+i]
        elif direction == BACK:
            for j in range(width):
                for i in range(height):
                    T[index(i+1, j+1, blockDimZ+1)] = gh[j*height+i]

    globals()['index'] = index
    globals()['compute_kernel_fast'] = compute_kernel_fast
    globals()['constrainBC_fast']    = constrainBC_fast
    globals()['fillGhostData']       = fillGhostData
    globals()['processGhosts_fast']  = processGhosts_fast


# This is just used to ensure more consistent benchmarking results,
# by compiling/loading all Numba functions on each PE once before any actual computations start
class NumbaPrecompiler(Chare):

    def __init__(self):
        #print("Numba warmup in PE", charm.myPe())
        make_numba_functions()
        size = (ro.blockDimX+2) * (ro.blockDimY+2) * (ro.blockDimZ+2)
        T = np.zeros(size, dtype='float64')
        compute_kernel_fast(10, T, T)
        constrainBC_fast(T)
        fillGhostData(T, T, T, T, T, T, T)
        processGhosts_fast(T, LEFT, 1, 1, T)
        del T


class Stencil(Chare):

    def __init__(self, sim_done_future):
        #print("Element " + str(self.thisIndex) + " created")

        arrSize = (ro.blockDimX+2) * (ro.blockDimY+2) * (ro.blockDimZ+2)
        if self.thisIndex == (0,0,0): print("array size=" + str(arrSize))
        self.temperature     = np.zeros(arrSize, dtype='float64')
        self.new_temperature = np.zeros(arrSize, dtype='float64')
        self.iterations = 0
        self.msgsRcvd = 0
        constrainBC_fast(self.temperature)

        # start measuring time
        if PRINT_ITERATIONS and self.thisIndex == (0,0,0): self.startTime = time.time()

        X,Y,Z = ro.num_chare_x, ro.num_chare_y, ro.num_chare_z
        i = self.thisIndex
        self.left_nb   = self.thisProxy[(i[0]-1)%X, i[1], i[2]]
        self.right_nb  = self.thisProxy[(i[0]+1)%X, i[1], i[2]]
        self.bottom_nb = self.thisProxy[i[0], (i[1]-1)%Y, i[2]]
        self.top_nb    = self.thisProxy[i[0], (i[1]+1)%Y, i[2]]
        self.front_nb  = self.thisProxy[i[0], i[1], (i[2]-1)%Z]
        self.back_nb   = self.thisProxy[i[0], i[1], (i[2]+1)%Z]
        self.me        = self.thisProxy[self.thisIndex]

        self.sim_done_future = sim_done_future

    def start(self):
        charm.LBTurnInstrumentOn()
        self.begin_iteration()

    def begin_iteration(self):
        self.iterations += 1
        blockDimX, blockDimY, blockDimZ = ro.blockDimX, ro.blockDimY, ro.blockDimZ

        # Copy different faces into messages
        leftGhost   = np.zeros((blockDimY*blockDimZ))
        rightGhost  = np.zeros((blockDimY*blockDimZ))
        topGhost    = np.zeros((blockDimX*blockDimZ))
        bottomGhost = np.zeros((blockDimX*blockDimZ))
        frontGhost  = np.zeros((blockDimX*blockDimY))
        backGhost   = np.zeros((blockDimX*blockDimY))

        fillGhostData(self.temperature, leftGhost, rightGhost, topGhost, bottomGhost, frontGhost, backGhost)

        self.left_nb.receiveGhosts(self.iterations, RIGHT, blockDimY, blockDimZ, leftGhost)   # Send my left face
        self.right_nb.receiveGhosts(self.iterations, LEFT, blockDimY, blockDimZ, rightGhost)  # Send my right face
        self.bottom_nb.receiveGhosts(self.iterations, TOP, blockDimX, blockDimZ, bottomGhost) # Send my bottom face
        self.top_nb.receiveGhosts(self.iterations, BOTTOM, blockDimX, blockDimZ, topGhost)    # Send my top face
        self.front_nb.receiveGhosts(self.iterations, BACK, blockDimX, blockDimY, frontGhost)  # Send my front face
        self.back_nb.receiveGhosts(self.iterations, FRONT, blockDimX, blockDimY, backGhost)   # Send my back face

    @when("self.iterations == iteration")
    def receiveGhosts(self, iteration, direction, height, width, gh):
        processGhosts_fast(self.temperature, direction, width, height, gh)
        self.msgsRcvd += 1
        if self.msgsRcvd == 6:
            self.msgsRcvd = 0
            self.me.check_and_compute()

    def check_and_compute(self):
        self.compute_kernel()

        # calculate error
        # not being done right now since we are doing a fixed no. of iterations

        self.temperature,self.new_temperature = self.new_temperature,self.temperature

        constrainBC_fast(self.temperature)

        if PRINT_ITERATIONS and self.thisIndex == (0,0,0):
            endTime = time.time()
            print("[" + str(self.iterations) + "] Time per iteration: " + str(endTime-self.startTime))

        if self.iterations == MAX_ITER:
            # notify main function that simulation is done
            self.contribute(None, None, self.sim_done_future)
        else:
            if PRINT_ITERATIONS and self.thisIndex == (0,0,0): self.startTime = time.time()
            if self.iterations == 1 or self.iterations % LBPERIOD_ITER == 0:
                self.AtSync()
            else:
                self.contribute(None, None, self.thisProxy.begin_iteration)

    # Check to see if we have received all neighbor values yet
    # If all neighbor values have been received, we update our values and proceed
    def compute_kernel(self):
        itno = int(math.ceil(float(self.iterations)/CHANGELOAD)) * 5
        i = self.thisIndex
        X,Y,Z = ro.num_chare_x, ro.num_chare_y, ro.num_chare_z
        idx = i[0] + i[1]*X + i[2]*X*Y
        numChares = X * Y * Z
        work = 100.0

        if (idx >= numChares*0.2) and (idx <= numChares*0.8):
            work = work * (float(idx)/numChares) + float(itno)
        else:
            work = 10.0

        compute_kernel_fast(int(work), self.new_temperature, self.temperature)

    def resumeFromSync(self):
        self.begin_iteration()


charm.start(main)
