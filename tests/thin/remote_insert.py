"""
ckInsert into an array from a process other than the one that created it (#320).

Run: python -m charmrun.start tests/thin/remote_insert.py +p1   (and +p2)

The main chare creates an empty 1D array on PE 0 and asks a group element on the
last PE to insert element 0 there, on its own PE, then send it a message. With 2
processes the insert happens in a process that never ran the array's ckNew; the
proxy must carry its chare class index for the binding to find the class.
Checks that the reply comes from the inserted element: index 0, the constructor
argument 42, and the PE it was inserted on.
"""
from charm4py import charm, Chare, Array, Group

class Item(Chare):
    def __init__(self, tag):
        self.tag = tag
    def hello(self, cb):
        cb((self.thisIndex[0], self.tag, charm.myPe()))

class Inserter(Chare):
    def insert(self, proxy, cb):
        proxy.ckInsert(0, args=[42], onPE=charm.myPe())
        proxy.ckDoneInserting()
        proxy[0].hello(cb)

def main(args):
    items = Array(Item, ndims=1)
    g = Group(Inserter)
    f = charm.Future()
    last = charm.numPes() - 1
    g[last].insert(items, f)
    reply = f.get()
    assert reply == (0, 42, last), 'unexpected reply ' + str(reply)
    print('remote_insert: reply', reply, 'numPes', charm.numPes())
    print('PASS')
    charm.exit()

charm.start(main)
