
This program solves the 2D wave equation over a grid, and displays a 2D
animation of wave propagation and interaction. An example screenshot is in
this directory.

This has been converted and adapted from ``examples/charm++/wave2d`` from the
Charm++ codebase. This version uses coroutines, channels and futures, and
displays the animation on process 0 using the Tk GUI toolkit.


Requirements
------------

- tkinter (Python interface to Tk GUI toolkit)

  - Windows: This package should come preinstalled with Python

  - Linux: There should be a python3-tk or similar package for your Linux distribution

- Python Image Library (PIL)
  *Make sure it is a recent version*. Can install with ``pip install pillow``

- Numba (see https://numba.pydata.org/ for installation instructions)

If tkinter or PIL is not found, the simulation will run but it won't show
any animation.

You can run without Numba, removing ``@numba.jit`` from the code, but the simulation
**will run VERY slowly**.


Usage
-----

wave2d.py [num_iterations] [max_framerate]

    num_iterations: number of iterations to run the simulation (default: 3000)

    max_framerate: maximum framerate (in frames per second) (default: 60)
    Set to -1 for unlimited (up to the speed allowed by your hardware)
