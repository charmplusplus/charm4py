===================
liveViz
===================

Introduction
-----

If array elements compute a small piece of a large 2D image, then these
image chunks can be combined across processors to form one large image
using the liveViz library. In other words, liveViz provides a way to
reduce 2D-image data, which combines small chunks of images deposited by
chares into one large image.

This visualization library follows the client server model. The server,
a parallel Charm4py program, does all image assembly, and opens a network
(CCS) socket which clients use to request and download images. The
client is a small Java program. A typical use of this is:

.. code-block:: bash

   	cd charm4py/examples/liveviz
   	python -m charmrun.start +p4 ++server ++local liveviz.py ++server-port 1234
   	~/ccs_tools/bin/liveViz localhost 1234

Use git to obtain a copy of ccs_tools (prior to using liveViz) and build
it by:

.. code-block:: bash

         cd ccs_tools;
         ant;

How to use liveViz with Charm4py program
---------------------------------------
A typical program provides a chare array with one entry method with the
following prototype:

.. code-block:: Python

     def functionName(self, request)

This entry method is supposed to deposit its (array element’s) chunk of
the image. This entry method has following structure:

.. code-block:: Python

     def functionName(self, request):
       #prepare image chunk

       liveviz.LiveViz.deposit(data, self, start_x, start_y, local_height, local_width, global_height, global_width)

Here, “local_width” and “local_height” are the size, in pixels, of this array
element’s portion of the image, contributed in data (described
below). This will show up on the client’s assembled image at 0-based
pixel (start_x,start_y). LiveViz combines image chunks by doing a saturating sum of
overlapping pixel values.

Format of deposit image
-----------------------

“data” is run of bytes representing a rectangular portion of the
image. This buffer represents image using a row-major format, so 0-based
pixel (x,y) (x increasing to the right, y increasing downward in typical
graphics fashion) is stored at array offset “x+y*width”.

If the image is gray-scale (as determined by liveVizConfig, below), each
pixel is represented by one byte. If the image is color, each pixel is
represented by 3 consecutive bytes representing red, green, and blue
intensity.

liveViz Initialization
----------------------

liveViz library needs to be initialized before it can be used for
visualization. For initialization follow the following steps from your
main chare:

#. Create your chare array (array proxy object ’a’) with the entry
   method ’functionName’ (described above).

#. Create a Config object (’cfg’). Config takes a number
   of parameters, as described below.

#. Call liveviz.LiveViz.init(cfg, a.functionName).

The liveviz.Config parameters are:

-  isColor, where “False” means a greyscale image (1
   byte per pixel) and “True”  means a color image (3 RGB
   bytes per pixel). This defaults to True.

-  The second parameter is the flag "isPush", which is passed to the
   client application. If set to True, the client will repeatedly
   request for images. When set to False the client will only request
   for images when its window is resized and needs to be updated. This also defaults to True.

Poll Mode
---------

In some cases you may want a server to deposit images only when it is
ready to do so. For this case the server will not register a callback
function that triggers image generation, but rather the server will
deposit an image at its convenience. For example a server may want to
create a movie or series of images corresponding to some timesteps in a
simulation. The server will have a timestep loop in which an array
computes some data for a timestep. At the end of each iteration the
server will deposit the image. The use of LiveViz’s Poll Mode supports
this type of server generation of images. To use poll mode, simply set 
poll = True during initialization.

.. code-block:: Python

   	liveviz.LiveViz.init(cfg, a.functionName, poll=True)

To deposit an image, the server just calls liveVizDeposit. The
server must take care not to generate too many images, before a client
requests them. Each server generated image is buffered until the client
can get the image. The buffered images will be stored in memory on
processor 0.

Sample liveViz and liveVizPoll servers are available at:

.. code-block:: none

              .../charm4py/examples/liveviz