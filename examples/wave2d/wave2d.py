
# This program solves the 2-d wave equation over a grid, displaying pretty results.
# See README.rst for more information.

from charm4py import charm, Chare, Array, coro, Channel, Future
import time
import math
import numpy as np
import numba
import random
try:
    import tkinter
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    import sys
    sys.argv += ['--NO-RENDER']


IMAGE_WIDTH, IMAGE_HEIGHT = 800, 699
CHARE_ARRAY_WIDTH, CHARE_ARRAY_HEIGHT = 4, 3
NUM_ITERATIONS = 3000
NUM_INITIAL_PERTURBATIONS = 5
LEFT, RIGHT, UP, DOWN = range(4)
MAX_FRAMERATE = 60  # in frames per second. set -1 for unlimited


class Main(Chare):

    def __init__(self, args):
        self.RENDER = True
        try:
            args.remove('--NO-RENDER')
            self.RENDER = False
        except ValueError:
            pass

        print('\nUsage: wave2d.py [num_iterations] [max_framerate])')
        global NUM_ITERATIONS, MAX_FRAMERATE
        if len(args) > 1:
            NUM_ITERATIONS = int(args[1])
        if len(args) > 2:
            MAX_FRAMERATE = int(args[2])

        print('Running wave2d on', charm.numPes(), 'processors for', NUM_ITERATIONS, 'iterations')
        print('Max framerate is', MAX_FRAMERATE, 'frames per second')

        self.count = 0  # tracks from how many workers I have received a subimage for this iteration
        programStartTime = frameStartTime = time.time()

        # Create new 2D array of worker chares
        array = Array(Wave, (CHARE_ARRAY_WIDTH, CHARE_ARRAY_HEIGHT))
        # tell all the worker chares to start the simulation
        array.work(self.thisProxy)

        if self.RENDER:
            tk = tkinter.Tk()
            self.frame = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = ImageTk.PhotoImage(self.frame)
            label_image = tkinter.Label(tk, image=img)
            label_image.pack()

        self.frameReady = Future()
        for i in range(NUM_ITERATIONS):
            self.frameReady.get()  # wait for the next frame
            if MAX_FRAMERATE > 0:
                elapsed = time.time() - frameStartTime
                if elapsed < 1/MAX_FRAMERATE:
                    # enforce framerate
                    charm.sleep(1/MAX_FRAMERATE - elapsed)
            if self.RENDER:
                fps = round(1/(time.time() - frameStartTime))
                # draw frames per second value on image
                d = ImageDraw.Draw(self.frame)
                d.text((10,10), str(fps) + ' fps', fill=(0,0,0,255))
                img = ImageTk.PhotoImage(self.frame)
                label_image.configure(image=img)
                label_image.image = img
                tk.update_idletasks()
                tk.update()

            # loop simulation every 1000 iterations
            reset = (i % 1000 == 0)
            frameStartTime = time.time()
            array.resume(reset)  # tell workers to resume
            self.frameReady = Future()

        print('Program Done!, Total time=', time.time() - programStartTime)
        exit()

    # every worker calls this method to deposit their subimage
    def depositSubImage(self, data, pos, img_size):
        self.count += 1
        if self.RENDER:
            self.frame.paste(Image.frombytes('RGB', img_size, data), box=pos)
        if self.count == CHARE_ARRAY_WIDTH * CHARE_ARRAY_HEIGHT:
            # received image data from all chares
            self.count = 0
            self.frameReady()  # signal main that the next frame is ready


class Wave(Chare):

    def setInitialConditions(self):
        # setup some initial pressure pertubations for timesteps t-1 and t
        self.pressure_new = np.zeros((self.myheight, self.mywidth))  # time t+1
        self.pressure     = np.zeros((self.myheight, self.mywidth))  # time t
        self.pressure_old = np.zeros((self.myheight, self.mywidth))  # time t-1
        init_pressure(NUM_INITIAL_PERTURBATIONS, IMAGE_WIDTH, IMAGE_HEIGHT,
                      self.mywidth, self.myheight, self.thisIndex,
                      self.pressure, self.pressure_old)

    def resume(self, reset=False):
        self.resumeFuture(reset)

    @coro
    def work(self, mainProxy):
        """ this is the main simulation loop for each chare """

        # size of my rectangular portion of the image
        self.mywidth = IMAGE_WIDTH // CHARE_ARRAY_WIDTH
        self.myheight = IMAGE_HEIGHT // CHARE_ARRAY_HEIGHT
        self.setInitialConditions()

        i = self.thisIndex
        X, Y = CHARE_ARRAY_WIDTH, CHARE_ARRAY_HEIGHT
        # establish a Channel with neighbor chares in the 2D grid
        left = Channel(self, remote=self.thisProxy[(i[0]-1)%X, i[1]])
        right = Channel(self, remote=self.thisProxy[(i[0]+1)%X, i[1]])
        top = Channel(self, remote=self.thisProxy[i[0], (i[1]-1)%Y])
        bottom = Channel(self, remote=self.thisProxy[i[0], (i[1]+1)%Y])

        width, height = self.mywidth, self.myheight
        # coordinate where my portion of the image is located
        sx = self.thisIndex[0] * width
        sy = self.thisIndex[1] * height
        # data will store my portion of the image
        data = np.zeros(width*height*3, dtype=np.uint8)
        buffers = [None] * 4

        # run simulation now
        while True:
            top_edge = self.pressure[[0],:].reshape(width)
            bottom_edge = self.pressure[[-1],:].reshape(width)
            left_edge = self.pressure[:,[0]].reshape(height)
            right_edge = self.pressure[:,[-1]].reshape(height)

            # send ghost values to neighbors
            left.send(RIGHT, left_edge)
            right.send(LEFT, right_edge)
            bottom.send(UP, bottom_edge)
            top.send(DOWN, top_edge)

            # receive ghost values from neighbors. iawait iteratively yields
            # channels as they become ready (have data to receive)
            for channel in charm.iwait((left, right, bottom, top)):
                side, ghost_values = channel.recv()
                buffers[side] = ghost_values

            check_and_compute(height, width,
                              buffers[LEFT], buffers[RIGHT], buffers[UP], buffers[DOWN],
                              self.pressure, self.pressure_old, self.pressure_new)

            # advance to next step by shifting the data back one step in time
            self.pressure_old, self.pressure, self.pressure_new = self.pressure, self.pressure_new, self.pressure_old

            # draw my part of the image, plus a nice 1 pixel border along my
            # right/bottom boundary
            fill_subimage(data, width, height, self.pressure)
            # provide my portion of the image to the mainchare
            mainProxy.depositSubImage(data, (sx, sy), (width, height))
            # wait for message from mainchare to resume simulation
            self.resumeFuture = Future()
            reset = self.resumeFuture.get()
            if reset:
                self.setInitialConditions()


@numba.jit(nopython=True, cache=False)
def check_and_compute(h, w, left, right, up, down,
                      pressure, pressure_old, pressure_new):
    for i in range(h):
        for j in range(w):
            # current time's pressures for neighboring array locations
            if j == 0: L = left[i]
            else: L = pressure[i,j-1]

            if j == w-1: R = right[i]
            else: R = pressure[i,j+1]

            if i == 0: U = up[j]
            else: U = pressure[i-1,j]

            if i == h-1: D = down[j]
            else: D = pressure[i+1,j]

            # current time's pressure for this array location
            curr = pressure[i,j]

            # previous time's pressure for this array location
            old = pressure_old[i,j]

            # compute the future time's pressure for this array location
            pressure_new[i,j] = 0.4*0.4*(L+R+U+D - 4.0*curr)-old+2.0*curr


@numba.jit(nopython=True, cache=False)
def fill_subimage(data, w, h, pressure):
    # set the output pixel values for my rectangle
    # Each RGB component is a uint8 that can have 256 possible values
    for i in range(h):
        for j in range(w):
            p = int(pressure[i,j])
            if p > 255: p = 255    # Keep values in valid range
            if p < -255: p = -255  # Keep values in valid range
            pos = 3*(i*w+j)
            if p > 0:  # Positive values are red
                data[pos:pos+3] = (255, 255-p, 255-p)
            else:  # Negative values are blue
                data[pos:pos+3] = (255+p, 255+p, 255)

    # Draw a green border on right and bottom of this chare array's pixel buffer.
    # This will overwrite some pressure values at these pixels.
    for i in range(h):
        pos = 3*(i*w+w-1)
        data[pos:pos+3] = (0, 255, 0)
    for i in range(w):
        pos = 3*((h-1)*w+i)
        data[pos:pos+3] = (0, 255, 0)


@numba.jit(nopython=True, cache=False)
def init_pressure(numInitialPerturbations, W, H, w, h, elemIdx, pressure, pressure_old):
    # force the same random numbers to be used for each chare array element
    random.seed(6)
    for s in range(numInitialPerturbations):
        # determine where to place a circle within the interior of the 2D domain
        radius = 20 + random.randint(0,32767) % 30
        xcenter = radius + random.randint(0,32767) % (W - 2*radius)
        ycenter = radius + random.randint(0,32767) % (H - 2*radius)
        # draw the circle
        for i in range(h):
            for j in range(w):
                # the coordinate in the global data array (not just in this chare's portion)
                globalx = elemIdx[0]*w + j
                globaly = elemIdx[1]*h + i
                distanceToCenter = math.sqrt((globalx-xcenter)**2 + (globaly-ycenter)**2)
                if distanceToCenter < radius:
                    rscaled = (distanceToCenter/radius)*3.0*3.14159/2.0  # ranges from 0 to 3pi/2
                    t = 700.0 * math.cos(rscaled)  # range won't exceed -700 to 700
                    pressure[i,j] = pressure_old[i,j] = t


charm.start(Main)
