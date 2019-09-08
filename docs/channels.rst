========
Channels
========

Channels are streamed connections or pipes between a pair of chares. They
simplify the expression of sends/receives from inside a coroutine without having to exit
the coroutine.

Creation
--------

To create a Channel:

* **Channel(chare, remote):**

    Establish a channel between *chare* and *remote*, returning a channel
    object.

    Note that *chare* is an actual chare object, not a proxy.
    *remote* is a proxy to the remote chare.

    Channels do not have to be created from coroutines, but they can only
    be used from coroutines.

    There is no restriction on the number of channels that a chare can
    establish, and it can establish multiple channels with the same remote.

Methods
-------

Channel objects have the following methods:

* **send(self, *args):**

    Send the arguments through the channel to the remote chare.

* **recv(self):**

    Receives arguments (unpacked) from the channel. Messages are received in
    order.

Example
-------

.. code-block:: python

    from charm4py import charm, Chare, Array, coro, Channel

    NUM_ITER = 100

    class A(Chare):

        def __init__(self, numchares):
            myidx = self.thisIndex[0]
            neighbors = []
            neighbors.append(self.thisProxy[(myidx + 1) % numchares])
            neighbors.append(self.thisProxy[(myidx - 1) % numchares])
            self.channels = []
            for nb in neighbors:
                self.channels.append(Channel(self, remote=nb))

        @coro
        def work(self):
            for i in range(NUM_ITER):
                for ch in self.channels:
                    ch.send(x, y, z)

                for ch in charm.iwait(self.channels):
                    x, y, z = ch.recv()
                    # ... do something with data ...

    def main(args):
        numchares = charm.numPes() * 8
        array = Array(A, numchares, args=[numchares])
        future = array.work(awaitable=True)
        future.get()
        exit()

    charm.start(main)
