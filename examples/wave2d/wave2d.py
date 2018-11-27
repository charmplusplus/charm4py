# This program solves the 2-d wave equation over a grid, displaying pretty
# results through liveViz
# The discretization used below is described in the accompanying paper.pdf
# Original author: Isaac Dooley (2008)
# Converted to Python from examples/charm++/wave2d in Charm++ codebase

from charm4py import charm, Chare, Array
from charm4py import readonlies as ro
import time
import math
import numpy as np
import numba
import random

TotalDataWidth, TotalDataHeight = 800, 699
chareArrayWidth, chareArrayHeight = 4, 3
default_total_iterations = 3000
numInitialPertubations = 5
LEFT, RIGHT, UP, DOWN = range(4)


class Main(Chare):
  def __init__(self, args):

    if len(args) <= 1:
      self.total_iterations = default_total_iterations
    else:
      self.total_iterations = int(args[1])

    self.iteration = self.count = 0
    self.programStartTime = self.periodStartTime = time.time()
    ro.mainProxy = self.thisProxy # store the main proxy

    print("Running wave2d on " + str(charm.numPes()) + " processors")

    # Create new array of worker chares
    ro.arrayProxy = Array(Wave, (chareArrayWidth, chareArrayHeight))
    # Start the computation
    ro.arrayProxy.begin_iteration(False)
    #charm.initLiveViz((TotalDataWidth,TotalDataHeight))
    #charm.liveViz.record("wave2d", 60)
    #charm.liveViz.stream(60)

  # Each worker calls this method
  def iterationCompleted(self, im, pos, imSize):
    self.count += 1
    #charm.liveViz.deposit([(pos,imSize,im)])
    if self.count == chareArrayWidth * chareArrayHeight:
      if self.iteration == self.total_iterations:
        print("Program Done!, Total time= " + str(time.time() - self.programStartTime))
        charm.printStats()
        exit()
      else:
        # Start the next iteration
        self.count = 0
        self.iteration += 1
        if self.iteration % 100 == 0:
          print("Completed " + str(self.iteration) + " iterations " + str(time.time() - self.periodStartTime))
          self.periodStartTime = time.time()
          #charm.liveViz.snapshot("examples/wave2d/wave2d-" + str(self.iteration) + ".png", "PNG")
        restart = (self.iteration % 1000 == 0)  # loop simulation every 1000 iterations
        ro.arrayProxy.begin_iteration(restart)

@numba.jit
def check_and_compute_fast(h,w,left,right,up,down,pressure,pressure_old,pressure_new):
  for i in range(h):
    for j in range(w):
      # Current time's pressures for neighboring array locations
      if j == 0: L = left[i]
      else: L = pressure[i,j-1]

      if j == w-1: R = right[i]
      else: R = pressure[i,j+1]

      if i == 0: U = up[j]
      else: U = pressure[i-1,j]

      if i == h-1: D = down[j]
      else: D = pressure[i+1,j]

      # Current time's pressure for this array location
      curr = pressure[i,j]

      # Previous time's pressure for this array location
      old = pressure_old[i,j]

      # Compute the future time's pressure for this array location
      pressure_new[i,j] = 0.4*0.4*(L+R+U+D - 4.0*curr)-old+2.0*curr

@numba.jit
def fillSubImage(data, w, h, pressure):
  # set the output pixel values for my rectangle
  # Each RGB component is a char which can have 256 possible values.
  for i in range(h):
    for j in range(w):
      p = int(pressure[i,j])
      if p > 255: p = 255    # Keep values in valid range
      if p < -255: p = -255  # Keep values in valid range
      pos = 3*(i*w+j)
      if p > 0: # Positive values are red
        data[pos:pos+3] = (255, 255-p, 255-p)
      else: # Negative values are blue
        data[pos:pos+3] = (255+p, 255+p, 255)

  # Draw a green border on right and bottom of this chare array's pixel buffer.
  # This will overwrite some pressure values at these pixels.
  for i in range(h):
    pos = 3*(i*w+w-1)
    data[pos:pos+3] = (0, 255, 0)
  for i in range(w):
    pos = 3*((h-1)*w+i)
    data[pos:pos+3] = (0, 255, 0)

@numba.jit
def initPressure(numInitialPertubations, W, H, w, h, elemIdx, pressure, pressure_old):
  random.seed(0) # Force the same random numbers to be used for each chare array element
  for s in range(numInitialPertubations):
    # Determine where to place a circle within the interior of the 2-d domain
    radius = 20 + random.randint(0,32767) % 30
    xcenter = radius + random.randint(0,32767) % (W - 2*radius)
    ycenter = radius + random.randint(0,32767) % (H - 2*radius)
    # Draw the circle
    for i in range(h):
      for j in range(w):
        globalx = elemIdx[0]*w + j #  The coordinate in the global data array (not just in this chare's portion)
        globaly = elemIdx[1]*h + i
        distanceToCenter = math.sqrt((globalx-xcenter)**2 + (globaly-ycenter)**2)
        if distanceToCenter < radius:
          rscaled = (distanceToCenter/radius)*3.0*3.14159/2.0 # ranges from 0 to 3pi/2
          t = 700.0 * math.cos(rscaled) # Range won't exceed -700 to 700
          pressure[i,j] = pressure_old[i,j] = t

class Wave(Chare):
  def __init__(self):
    self.mywidth = TotalDataWidth // chareArrayWidth
    self.myheight = TotalDataHeight // chareArrayHeight
    self.buffers = [None] * 4
    self.messages_due = 4
    self.InitialConditions()

  # Setup some Initial pressure pertubations for timesteps t-1 and t
  def InitialConditions(self):
    self.pressure_new  = np.zeros((self.myheight, self.mywidth)) # time t+1
    self.pressure      = np.zeros((self.myheight, self.mywidth)) # time t
    self.pressure_old  = np.zeros((self.myheight, self.mywidth)) # time t-1
    initPressure(numInitialPertubations, TotalDataWidth, TotalDataHeight, self.mywidth, self.myheight, self.thisIndex, self.pressure, self.pressure_old)

  def begin_iteration(self, restart):
    if restart: self.InitialConditions()
    top_edge = self.pressure[[0],:].reshape(self.mywidth)
    bottom_edge = self.pressure[[-1],:].reshape(self.mywidth)
    left_edge = self.pressure[:,[0]].reshape(self.myheight)
    right_edge = self.pressure[:,[-1]].reshape(self.myheight)

    X,Y = chareArrayWidth, chareArrayHeight
    i = self.thisIndex
    self.thisProxy[(i[0]-1)%X, i[1]].recvGhosts(RIGHT, left_edge) # Send my left edge
    self.thisProxy[(i[0]+1)%X, i[1]].recvGhosts(LEFT, right_edge) # Send my right edge
    self.thisProxy[i[0], (i[1]-1)%Y].recvGhosts(DOWN, top_edge) # Send my top edge
    self.thisProxy[i[0], (i[1]+1)%Y].recvGhosts(UP, bottom_edge) # Send my bottom edge

  def recvGhosts(self, whichSide, ghost_values):
    self.buffers[whichSide] = ghost_values
    self.check_and_compute()

  def check_and_compute(self):
    self.messages_due -= 1
    if self.messages_due == 0:
      check_and_compute_fast(self.myheight,self.mywidth,self.buffers[LEFT],self.buffers[RIGHT],self.buffers[UP],self.buffers[DOWN],self.pressure,self.pressure_old,self.pressure_new)
      # Advance to next step by shifting the data back one step in time
      self.pressure_old, self.pressure, self.pressure_new = self.pressure, self.pressure_new, self.pressure_old
      self.messages_due = 4
      self.thisProxy[self.thisIndex].requestNextFrame()
      #self.requestNextFrame()

  # provide my portion of the image to liveViz
  def requestNextFrame(self):
    w,h = self.mywidth, self.myheight # Size of my rectangular portion of the image
    data = np.zeros(w*h*3, dtype=np.uint8)
    # Draw my part of the image, plus a nice 1px border along my right/bottom boundary
    fillSubImage(data, w, h, self.pressure)
    sx = self.thisIndex[0] * w # where my portion of the image is located
    sy = self.thisIndex[1] * h
    ro.mainProxy.iterationCompleted(data, (sx,sy), (w,h))


charm.start(Main)
