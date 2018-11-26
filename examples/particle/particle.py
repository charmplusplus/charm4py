from charm4py import charm, Chare, Array, when, Reducer
from charm4py import readonlies as ro
import time
import random
import math
import array

"""
    Simple particle simulation where a 2D box is decomposed into cells.
    Each cell has a set of particles; in each iteration particles move randomly.
    Particles can move from one cell to another.
    Each cell is a Chare, and the 2D grid is a 2D array of Cells.
"""

import sys
sys.argv += ['+LBPeriod', '0.001', '+LBCommOff']

NUM_ITER = 100
MAX_START_PARTICLES_PER_CELL = 5000
SIM_BOX_SIZE = 100.0


class Particle(object):

    def __init__(self, x, y):
        self.coords = [x, y]

    def perturb(self):
        for i in range(len(self.coords)):
            self.coords[i] += random.uniform(-ro.cellSize[i]*0.3, ro.cellSize[i]*0.3)
            if self.coords[i] > SIM_BOX_SIZE:
                self.coords[i] -= SIM_BOX_SIZE
            elif self.coords[i] < 0:
                self.coords[i] += SIM_BOX_SIZE


class Cell(Chare):

    def __init__(self, simDoneFuture):
        # store future to notify main function when simulation is done
        self.simDoneFuture = simDoneFuture
        self.iteration = -1
        self.particles = []
        self.msgsRcvd = 0
        # create particles in this cell
        lo_x = self.thisIndex[0] * ro.cellSize[0]
        lo_y = self.thisIndex[1] * ro.cellSize[1]
        for i in range(self.getNumParticles(ro.arrayDims)):
            self.particles.append(Particle(random.uniform(lo_x, lo_x+ro.cellSize[0]-0.001),
                                           random.uniform(lo_y, lo_y+ro.cellSize[1]-0.001)))
        # obtain list of my neighbors in 2D cell grid
        self.neighbors = self.getNbIndexes()

    def getNumParticles(self, dims):
        # assigns more particles to cells closer to center
        d = math.sqrt((self.thisIndex[0]-dims[0]/2)**2 + (self.thisIndex[1]-dims[1]/2)**2)
        max_d = math.sqrt((dims[0]/2)**2 + (dims[1]/2)**2) + 0.001
        return int((1-(d/max_d)) * MAX_START_PARTICLES_PER_CELL)

    def run(self):
        self.iteration += 1
        outgoingParticles = {nb: array.array('d') for nb in self.neighbors}
        i = 0
        while i < len(self.particles):
            p = self.particles[i]
            p.perturb()
            dest_cell = (int(p.coords[0] / ro.cellSize[0]), int(p.coords[1] / ro.cellSize[1]))
            if dest_cell != self.thisIndex:
                outgoingParticles[dest_cell].append(p.coords[0])
                outgoingParticles[dest_cell].append(p.coords[1])
                self.particles[i] = self.particles[-1]
                self.particles.pop()
            else:
               i += 1

        for nb in self.neighbors:
            self.thisProxy[nb].updateNeighbor(self.iteration, outgoingParticles[nb])

    @when("self.iteration == iter")
    def updateNeighbor(self, iter, particles):
        self.particles += [Particle(float(particles[i]), float(particles[i+1])) for i in range(0,len(particles),2)]
        self.msgsRcvd += 1
        if self.msgsRcvd == len(self.neighbors):
            self.msgsRcvd = 0
            self.contribute(len(self.particles), Reducer.max, self.thisProxy[(0,0)].collectMax)
            if self.iteration >= NUM_ITER:
              self.contribute(None, None, self.simDoneFuture) # simulation done
            elif self.iteration == 1 or self.iteration % 15 == 0:
              self.AtSync() # do load balancing
            else:
              self.run() # go to next iteration

    def resumeFromSync(self):
        self.run()

    def collectMax(self, max_particles):
        if self.iteration % 10 == 0:
            print("Max particles= " + str(max_particles))

    def getNbIndexes(self):
        nbs = set()
        x,y = self.thisIndex
        nb_x_coords = [(x-1)%ro.arrayDims[0], x, (x+1)%ro.arrayDims[0]]
        nb_y_coords = [(y-1)%ro.arrayDims[1], y, (y+1)%ro.arrayDims[1]]
        for nb_x in nb_x_coords:
            for nb_y in nb_y_coords:
                if (nb_x,nb_y) != self.thisIndex: nbs.add((nb_x,nb_y))
        return list(nbs)


def main(args):

    # put sim parameters in readonlies container. objects stored in this container are
    # broadcasted to all other processes, and can be accessed using same container in remote process
    if len(args) == 3:
        ro.arrayDims = (int(args[1]), int(args[2]))
    else:
        ro.arrayDims = (6, 3)  # default: 2D chare array of 6x3 cells
    ro.cellSize = (SIM_BOX_SIZE / ro.arrayDims[0], SIM_BOX_SIZE / ro.arrayDims[1])

    # create 2D Cell chare array and start simulation
    simDone = charm.createFuture()
    cells = Array(Cell, ro.arrayDims, args=[simDone]) # array creation happens asynchronously
    t0 = time.time()
    cells.run()
    # wait for simulation to complete
    simDone.get()
    print("Particle simulation done, elapsed time=", round(time.time() - t0, 3), "secs")
    exit()


charm.start(main)
