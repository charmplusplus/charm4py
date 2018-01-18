from charmpy import charm, Mainchare, Array, CkExit, when, Reducer
from charmpy import readonlies as ro
import random
import math
import array

NUM_ITER = 100
MAX_START_PARTICLES_PER_CELL = 5000
SIM_BOX_SIZE = 100.0

class Particle:
  def __init__(self, x, y):
    self.coords = [x,y]
  def perturb(self):
    for i in range(len(self.coords)):
      self.coords[i] += random.uniform(-ro.cellSize[i]*0.3, ro.cellSize[i]*0.3)
      if self.coords[i] > SIM_BOX_SIZE: self.coords[i] -= SIM_BOX_SIZE
      elif self.coords[i] < 0: self.coords[i] += SIM_BOX_SIZE

class Main(Mainchare):
  def __init__(self, args):
    super(Main,self).__init__()

    if len(args) == 3: ro.arrayDims = (int(args[1]), int(args[2]))
    else: ro.arrayDims = (6,3)  # default: 2D chare array 6x3
    ro.cellSize = (SIM_BOX_SIZE / ro.arrayDims[0], SIM_BOX_SIZE / ro.arrayDims[1])
    ro.mainProxy = self.thisProxy
    ro.cellProxy = charm.CellProxy.ckNew(ro.arrayDims)
    ro.cellProxy.run()

  def collectMax(self, max_particles):
    print("Max particles= " + str(max_particles))

  def done(self):
    CkExit()

def getNumParticles(pos, dims):  # assigns more particles to cells closer to center
  d = math.sqrt((pos[0]-dims[0]/2)**2 + (pos[1]-dims[1]/2)**2)
  max_d = math.sqrt((dims[0]/2)**2 + (dims[1]/2)**2) + 0.001
  return int((1-(d/max_d)) * MAX_START_PARTICLES_PER_CELL)

class Cell(Array):
  def __init__(self):
    super(Cell,self).__init__()
    self.iteration = 0
    self.particles = []
    self.msgsRcvd = 0
    lo_x = self.thisIndex[0] * ro.cellSize[0]
    lo_y = self.thisIndex[1] * ro.cellSize[1]
    for i in range(getNumParticles(self.thisIndex, ro.arrayDims)):
      self.particles.append(Particle(random.uniform(lo_x, lo_x+ro.cellSize[0]-0.001),
                                     random.uniform(lo_y, lo_y+ro.cellSize[1]-0.001)))
    self.neighbors = self.getNbIndexes() # list of neighbor indexes as tuples

  def run(self):
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
      ro.cellProxy[nb].updateNeighbor(self.iteration, outgoingParticles[nb])

  @when("iteration")
  def updateNeighbor(self, iter, particles):
    self.particles += [Particle(float(particles[i]), float(particles[i+1])) for i in range(0,len(particles),2)]
    self.msgsRcvd += 1
    if self.msgsRcvd == len(self.neighbors):
      self.msgsRcvd = 0
      self.contribute(len(self.particles), Reducer.max, ro.mainProxy.collectMax)
      self.iteration += 1
      if self.iteration >= NUM_ITER: self.contribute(None, None, ro.mainProxy.done)
      elif self.iteration % 10 == 2: self.AtSync() # do load balancing
      else: self.run() # keep running

  def resumeFromSync(self): self.run()

  def getNbIndexes(self):
    nbs = set()
    x,y = self.thisIndex
    nb_x_coords = [(x-1)%ro.arrayDims[0], x, (x+1)%ro.arrayDims[0]]
    nb_y_coords = [(y-1)%ro.arrayDims[1], y, (y+1)%ro.arrayDims[1]]
    for nb_x in nb_x_coords:
      for nb_y in nb_y_coords:
        if (nb_x,nb_y) != self.thisIndex: nbs.add((nb_x,nb_y))
    return list(nbs)

# ------ start charm -------

charm.start([Main,Cell])
