# Port of examples/charm++/load_balancing/stencil3d from Charm++ codebase
# This version doesn't use NumPy. It runs fairly well on PyPy, but NumPy+Numba version
# (stencil3d_numba.py) performs much better

from charm4py import charm, Chare, Array, when
from charm4py import readonlies as ro
import time
import math

import sys
sys.argv += ['+LBPeriod', '0.001', '+LBOff', '+LBCommOff']


MAX_ITER = 60
LBPERIOD_ITER = 10     # LB is called every LBPERIOD_ITER number of program iterations
CHANGELOAD = 30
LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK = range(6)
DIVIDEBY7 = 0.14285714285714285714


def index(a,b,c):
    return (a + b*(ro.blockDimX+2) + c*(ro.blockDimX+2)*(ro.blockDimY+2))


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

    # Create new array of worker chares
    sim_done = charm.createFuture()
    array = Array(Stencil, (ro.num_chare_x, ro.num_chare_y, ro.num_chare_z), args=[sim_done])
    charm.awaitCreation(array)

    print("Starting simulation\n\nNOTE: stencil3d_numba.py version performs much better than this one\n")
    initTime = time.time()
    array.start()
    sim_done.get() # wait until simulation completes
    totalTime = time.time() - initTime
    print(MAX_ITER, "iterations completed, total time=", round(totalTime,3), "secs, time per iteration (ms) =", round(totalTime / MAX_ITER * 1000, 3))
    exit()


class Stencil(Chare):

    def __init__(self, sim_done_future):
        #print("Element " + str(self.thisIndex) + " created")

        # NOTE: this uses lists for double arrays, better to use numpy (see stencil3d_numba.py)
        arrSize = (ro.blockDimX+2) * (ro.blockDimY+2) * (ro.blockDimZ+2)
        self.temperature     = [0.0] * arrSize
        self.new_temperature = [0.0] * arrSize
        self.iterations = 0
        self.msgsRcvd = 0
        self.constrainBC()

        # start measuring time
        if self.thisIndex == (0,0,0): self.startTime = time.time()
        self.sim_done_future = sim_done_future

    def start(self):
        charm.LBTurnInstrumentOn()
        self.begin_iteration()

    def begin_iteration(self):
        self.iterations += 1
        blockDimX, blockDimY, blockDimZ = ro.blockDimX, ro.blockDimY, ro.blockDimZ

        # Copy different faces into messages
        leftGhost   = [0.0] * (blockDimY * blockDimZ)
        rightGhost  = [0.0] * (blockDimY * blockDimZ)
        topGhost    = [0.0] * (blockDimX * blockDimZ)
        bottomGhost = [0.0] * (blockDimX * blockDimZ)
        frontGhost  = [0.0] * (blockDimX * blockDimY)
        backGhost   = [0.0] * (blockDimX * blockDimY)

        for k in range(blockDimZ):
            for j in range(blockDimY):
                leftGhost[k*blockDimY+j] = self.temperature[index(1, j+1, k+1)]
                rightGhost[k*blockDimY+j] = self.temperature[index(blockDimX, j+1, k+1)]

        for k in range(blockDimZ):
            for i in range(blockDimX):
                topGhost[k*blockDimX+i] = self.temperature[index(i+1, 1, k+1)]
                bottomGhost[k*blockDimX+i] = self.temperature[index(i+1, blockDimY, k+1)]

        for j in range(blockDimY):
            for i in range(blockDimX):
                frontGhost[j*blockDimX+i] = self.temperature[index(i+1, j+1, 1)];
                backGhost[j*blockDimX+i] = self.temperature[index(i+1, j+1, blockDimZ)];

        X,Y,Z = ro.num_chare_x, ro.num_chare_y, ro.num_chare_z
        i = self.thisIndex

        # Send my left face
        self.thisProxy[(i[0]-1)%X, i[1], i[2]].receiveGhosts(self.iterations, RIGHT, blockDimY, blockDimZ, leftGhost)
        # Send my right face
        self.thisProxy[(i[0]+1)%X, i[1], i[2]].receiveGhosts(self.iterations, LEFT, blockDimY, blockDimZ, rightGhost)
        # Send my bottom face
        self.thisProxy[i[0], (i[1]-1)%Y, i[2]].receiveGhosts(self.iterations, TOP, blockDimX, blockDimZ, bottomGhost)
        # Send my top face
        self.thisProxy[i[0], (i[1]+1)%Y, i[2]].receiveGhosts(self.iterations, BOTTOM, blockDimX, blockDimZ, topGhost)
        # Send my front face
        self.thisProxy[i[0], i[1], (i[2]-1)%Z].receiveGhosts(self.iterations, BACK, blockDimX, blockDimY, frontGhost)
        # Send my back face
        self.thisProxy[i[0], i[1], (i[2]+1)%Z].receiveGhosts(self.iterations, FRONT, blockDimX, blockDimY, backGhost)

    @when("self.iterations == iteration")
    def receiveGhosts(self, iteration, direction, height, width, gh):
        self.processGhosts(direction, height, width, gh)
        self.msgsRcvd += 1
        if self.msgsRcvd == 6:
            self.msgsRcvd = 0
            self.thisProxy[self.thisIndex].check_and_compute()

    def processGhosts(self, direction, height, width, gh):
        blockDimX, blockDimY, blockDimZ = ro.blockDimX, ro.blockDimY, ro.blockDimZ
        def index2(a,b,c):
            return (a + b*(blockDimX+2) + c*(blockDimX+2)*(blockDimY+2))

        if direction == LEFT:
            for k in range(width):
                for j in range(height):
                    self.temperature[index2(0, j+1, k+1)] = gh[k*height+j]
        elif direction == RIGHT:
            for k in range(width):
                for j in range(height):
                    self.temperature[index2(blockDimX+1, j+1, k+1)] = gh[k*height+j]
        elif direction == BOTTOM:
            for k in range(width):
                for i in range(height):
                    self.temperature[index2(i+1, 0, k+1)] = gh[k*height+i]
        elif direction == TOP:
            for k in range(width):
                for i in range(height):
                    self.temperature[index2(i+1, blockDimY+1, k+1)] = gh[k*height+i]
        elif direction == FRONT:
            for j in range(width):
                for i in range(height):
                    self.temperature[index2(i+1, j+1, 0)] = gh[j*height+i]
        elif direction == BACK:
            for j in range(width):
                for i in range(height):
                    self.temperature[index2(i+1, j+1, blockDimZ+1)] = gh[j*height+i]
        else:
            charm.abort("ERROR\n");

    def check_and_compute(self):
        self.compute_kernel()

        # calculate error
        # not being done right now since we are doing a fixed no. of iterations

        self.temperature,self.new_temperature = self.new_temperature,self.temperature

        self.constrainBC()

        if self.thisIndex == (0,0,0):
            endTime = time.time()
            print("[" + str(self.iterations) + "] Time per iteration: " + str(endTime-self.startTime))

        if self.iterations == MAX_ITER:
            self.contribute(None, None, self.sim_done_future)
        else:
            if self.thisIndex == (0,0,0): self.startTime = time.time()
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

        blockDimX, blockDimY, blockDimZ = ro.blockDimX, ro.blockDimY, ro.blockDimZ
        for w in range(int(work)):
            for k in range(1, blockDimZ+1):
                for j in range(1, blockDimY+1):
                    for i in range(1, blockDimX+1):
                        # update my value based on the surrounding values
                        self.new_temperature[index(i, j, k)] = (self.temperature[index(i-1, j, k)] \
                            +  self.temperature[index(i+1, j, k)] \
                            +  self.temperature[index(i, j-1, k)] \
                            +  self.temperature[index(i, j+1, k)] \
                            +  self.temperature[index(i, j, k-1)] \
                            +  self.temperature[index(i, j, k+1)] \
                            +  self.temperature[index(i, j, k)] ) \
                            *  DIVIDEBY7

    # Enforce some boundary conditions
    def constrainBC(self):
        blockDimX, blockDimY, blockDimZ = ro.blockDimX, ro.blockDimY, ro.blockDimZ
        T = self.temperature
        # Heat left, top and front faces of each chare's block
        for k in range(1,blockDimZ+1):
            for i in range(1,blockDimX+1):
                T[index(i, 1, k)] = 255.0

        for k in range(1, blockDimZ+1):
            for j in range(1, blockDimY+1):
                T[index(1, j, k)] = 255.0

        for j in range(1, blockDimY+1):
            for i in range(1, blockDimX+1):
                T[index(i, j, 1)] = 255.0

    def resumeFromSync(self):
        self.begin_iteration()


charm.start(main)
