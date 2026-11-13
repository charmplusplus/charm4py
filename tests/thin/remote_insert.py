"""
ckInsert into an array from a process other than the one that created it (#320).

Run: python -m charmrun.start tests/thin/remote_insert.py +p1   (and +p2)

The main chare creates empty arrays on PE 0 and asks a group element on the last
PE to insert into them, then send each new element a message. With 2 processes
the inserts happen in a process that never ran the arrays' ckNew; the proxy must
carry its chare class index for the binding to find the class. Paths covered:
the proxy as a direct argument (PROXY slot) with onPE=myPe and onPE=-1 (default
placement), the proxy inside a dict (it arrives through pickle), and a second
class with a 2-D index. Each reply carries the index, constructor argument and PE.
"""
from charm4py import charm, Chare, Array, Group

class Item(Chare):
    def __init__(self, tag):
        self.tag = tag
    def hello(self, cb):
        cb((tuple(self.thisIndex), self.tag, charm.myPe()))

class Item2(Item):  # a second class: a wrong class index would reply with 'Item'
    def hello(self, cb):
        cb((tuple(self.thisIndex), self.tag, charm.myPe(), 'Item2'))

class Inserter(Chare):
    def insert(self, proxy, idx, tag, onPE, cb):
        proxy.ckInsert(idx, args=[tag], onPE=charm.myPe() if onPE == 'mine' else onPE)
        proxy.ckDoneInserting()
        proxy[idx].hello(cb)
    def insert_boxed(self, box, cb):
        self.insert(box['proxy'], 2, 'boxed', 'mine', cb)

def check(name, reply, expect):
    if reply != expect:
        charm.abort('remote_insert: %s: expected %s, got %s' % (name, expect, reply))
    print('remote_insert: PASS', name, reply)

def main(args):
    last = charm.numPes() - 1
    items = Array(Item, ndims=1)
    items2 = Array(Item2, ndims=2)
    g = Group(Inserter)

    f = charm.Future(); g[last].insert(items, 0, 42, 'mine', f)
    check('direct onPE=myPe', f.get(), ((0,), 42, last))

    f = charm.Future(); g[last].insert(items, 1, 43, -1, f)
    reply = f.get()  # default placement: the PE is the map's choice, any valid one
    check('direct onPE=-1', reply, ((1,), 43, reply[2] if 0 <= reply[2] <= last else 'a PE in range'))

    f = charm.Future(); g[last].insert_boxed({'proxy': items}, f)
    check('proxy in dict (pickle)', f.get(), ((2,), 'boxed', last))

    f = charm.Future(); g[last].insert(items2, (1, 2), 'two-d', 'mine', f)
    check('Item2 2-D index', f.get(), ((1, 2), 'two-d', last, 'Item2'))

    print('remote_insert: numPes', charm.numPes(), 'PASS')
    charm.exit()

charm.start(main)
