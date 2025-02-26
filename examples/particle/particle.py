from charm4py import charm, Chare, Array, Reducer, Future, coro, Channel
import time
import random
import math
import array
import sys

sys.argv += ['+LBCommOff', '+LBObjOnly']

NUM_ITER = 100
SIM_BOX_SIZE = 100.0

class Particle(object):

    def __init__(self, x, y):
        self.coords = [x, y]  # coordinate of this particle in the 2D space

    def perturb(self, cellsize):
        for i in range(len(self.coords)):
            self.coords[i] += random.uniform(-cellsize[i]*0.1, cellsize[i]*0.1)
            # if particle goes out of bounds of the simulation space, appear on the other side
            if self.coords[i] > SIM_BOX_SIZE:
                self.coords[i] -= SIM_BOX_SIZE
            elif self.coords[i] < 0:
                self.coords[i] += SIM_BOX_SIZE


class Cell(Chare):

    def __init__(self, array_dims, max_particles_per_cell_start, sim_done_future):
        self.sim_done_future = sim_done_future
        self.iteration = 0
        cellsize = (SIM_BOX_SIZE / array_dims[0], SIM_BOX_SIZE / array_dims[1])
        self.cellsize = cellsize

        self.particles = []
        N = self.getInitialNumParticles(array_dims, max_particles_per_cell_start, cellsize)
        lo_x = self.thisIndex[0] * cellsize[0]  # x coordinate of lower left corner of my cell
        lo_y = self.thisIndex[1] * cellsize[1]  # y coordinate of lower left corner of my cell
        for _ in range(N):
            self.particles.append(Particle(random.uniform(lo_x, lo_x + cellsize[0] - 0.001),
                                           random.uniform(lo_y, lo_y + cellsize[1] - 0.001)))

        self.neighbor_indexes = self.getNbIndexes(array_dims)
        self.neighbors = [Channel(self, remote=self.thisProxy[idx]) for idx in self.neighbor_indexes]

    def getInitialNumParticles(self, dims, max_particles, cellsize):
        grid_center = (SIM_BOX_SIZE / 2, SIM_BOX_SIZE / 2)
        cell_center = (self.thisIndex[0] * cellsize[0] + cellsize[0] / 2,
                       self.thisIndex[1] * cellsize[1] + cellsize[1] / 2)
        dist = math.sqrt((cell_center[0] - grid_center[0])**2 + (cell_center[1] - grid_center[1])**2)
        if dist <= SIM_BOX_SIZE / 5:
            return max_particles
        else:
            return 0

    def getNbIndexes(self, arrayDims):
        nbs = set()
        x, y = self.thisIndex
        nb_x_coords = [(x-1)%arrayDims[0], x, (x+1)%arrayDims[0]]
        nb_y_coords = [(y-1)%arrayDims[1], y, (y+1)%arrayDims[1]]
        for nb_x in nb_x_coords:
            for nb_y in nb_y_coords:
                if (nb_x, nb_y) != self.thisIndex:
                    nbs.add((nb_x, nb_y))
        return list(nbs)

    def getNumParticles(self):
        return len(self.particles)

    @coro
    def run(self):
        """ this is the simulation loop of each cell """
        cellsize = self.cellsize
        iteration = 0
        for iteration in range(NUM_ITER):
            outgoingParticles = {nb_idx: array.array('d') for nb_idx in self.neighbor_indexes}
            i = 0
            while i < len(self.particles):
                p = self.particles[i]
                p.perturb(cellsize)
                dest_cell = (int(p.coords[0] / cellsize[0]), int(p.coords[1] / cellsize[1]))
                if dest_cell != self.thisIndex:
                    outgoingParticles[dest_cell].extend(p.coords)
                    self.particles[i] = self.particles[-1]
                    self.particles.pop()
                else:
                    i += 1

            for i, channel in enumerate(self.neighbors):
                channel.send(outgoingParticles[self.neighbor_indexes[i]])

            for channel in charm.iwait(self.neighbors):
                incoming = channel.recv()
                self.particles += [Particle(float(incoming[i]),
                                            float(incoming[i+1])) for i in range(0, len(incoming), 2)]

            if iteration % 10 == 0:
                self.reduce(self.thisProxy[(0,0)].reportMax, len(self.particles), Reducer.max)

            if iteration > 0 and iteration % 20 == 0:
                self.AtSyncAndWait()
                # here is where we would have to re-initialize self things, like neighbors
        self.reduce(self.sim_done_future)

    # def resumeFromSync(self):
        # self.thisProxy[self.thisIndex].run()

    def reportMax(self, max_particles):
        print('Max particles per cell= ' + str(max_particles))


def main(args):
    print('\nUsage: particle.py [num_chares_x num_chares_y] [max_particles_per_cell_start]')
    if len(args) >= 3:
        array_dims = (int(args[1]), int(args[2]))
    else:
        array_dims = (1, 2)  # default: 2D chare array of 8 x 4 cells
    if len(args) == 4:
        max_particles_per_cell_start = int(args[3])
    else:
        max_particles_per_cell_start = 10000

    print('\nCell array size:', array_dims[0], 'x', array_dims[1], 'cells')
    chares_x = 10
    chares_y = 10

    sim_done = Future()
    cells = Array(Cell, (chares_x, chares_y),
                  args=[(chares_x, chares_y), max_particles_per_cell_start, sim_done],
                  useAtSync=True)
    num_particles_per_cell = cells.getNumParticles(ret=True).get()
    print('Total particles created:', sum(num_particles_per_cell))
    print('Initial conditions:\n\tmin particles per cell:', min(num_particles_per_cell),
          '\n\tmax particles per cell:', max(num_particles_per_cell))
    print('\nStarting simulation')
    t0 = time.time()
    cells.run()  # this is a broadcast
    sim_done.get()
    print('Particle simulation done, elapsed time=', round(time.time() - t0, 3), 'secs')
    exit()


charm.start(main)
