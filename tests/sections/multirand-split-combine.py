from charm4py import charm, Chare, Group, Array, Reducer, Future
import random
import numpy


charm.options.local_msg_buf_size = 10000


NUM_ARRAYS = 10
NUM_GROUPS = 10
NUM_GEN = 300
DATA_VERIFY = 0
NUM_ITER = 1
VERBOSE = False
insections = None


def mygather(contribs):
    result = []
    for c in contribs:
        result += c
    return result


Reducer.addReducer(mygather)


class Test(Chare):

    def __init__(self, cid):
        self.cid = cid
        if isinstance(self.thisIndex, int):
            self.idx = (self.thisIndex,)
        else:
            self.idx = self.thisIndex
        self.secProxies = {}  # sid -> proxy

    def getIds(self, f):
        self.contribute([(self.cid, self.idx)], Reducer.mygather, f)

    def recvSecProxy(self, sid, proxy):
        assert sid not in self.secProxies
        self.secProxies[sid] = proxy

    def recvBcast00(self, f, secProxy):
        self.contribute([(self.cid, self.idx)], Reducer.mygather, f, secProxy)

    def recvBcast(self, x, f, sid=None):
        assert x == DATA_VERIFY
        section = None
        if sid is not None:
            section = self.secProxies[sid]
        self.contribute([(self.cid, self.idx)], Reducer.mygather, f, section)


class Collection(object):

    def __init__(self, elems, proxy):
        self.elems = set(elems)  # set of (cid, idx)
        assert len(self.elems) == len(elems)
        self.proxy = proxy

    def addElems(self, elems):
        # collections resulting from combining sections could get the same
        # element from different sections
        self.elems.update(elems)

    def verify(self, result):
        if set(result) != self.elems:
            print('self.elems=', self.elems)
            print('result=', result)
            raise Exception

    def split(self, N):
        insections = {}  # (cid, idx) -> [section_num]
        if random.random() <= 0.5:
            # disjoint
            elems = list(self.elems)
            random.shuffle(elems)
            sections = partition(elems, N)
            assert len(sections) == N
            for i, section in enumerate(sections):
                for cid, idx in section:
                    assert (cid, idx) not in insections
                    insections[(cid, idx)] = i
        else:
            # chance of non disjoint sections
            sections = []
            elems = list(self.elems)
            for i in range(N):
                elems_sec = random.sample(elems, max(len(elems) // N, 1))
                sections.append(elems_sec)
                for cid, idx in elems_sec:
                    if (cid, idx) not in insections:
                        insections[(cid, idx)] = []
                    insections[(cid, idx)].append(i)
        charm.thisProxy.updateGlobals({'insections': insections}, awaitable=True).get()
        assert len(sections) == N
        for section in sections:
            assert len(section) > 0
        proxies = charm.split(self.proxy, N, inSections)
        assert len(proxies) == N
        collections = [Collection(sections[i], proxies[i]) for i in range(N)]
        return collections


def partition(elems, N):
    num_elems = len(elems)
    return [elems[i*num_elems // N: (i+1)*num_elems // N]
            for i in range(N)]


def inSections(obj):
    if (obj.cid, obj.idx) in insections:
        return insections[(obj.cid, obj.idx)]
    else:
        return []


def main(args):
    collections = []
    cid = 0  # collection ID

    for _ in range(NUM_ARRAYS):
        proxy = Array(Test, random.randint(1, charm.numPes() * 10), args=[cid])
        f = Future()
        proxy.getIds(f)
        collections.append(Collection(f.get(), proxy))
        cid += 1

    for _ in range(NUM_GROUPS):
        proxy = Group(Test, args=[cid])
        f = Future()
        proxy.getIds(f)
        collections.append(Collection(f.get(), proxy))
        cid += 1

    sections_split = 0
    sections_combined = 0

    for i in range(NUM_GEN):
        if i < 20 or random.random() <= 0.5:
            # split
            while True:
                c = random.choice(collections)
                if len(c.elems) > 1:
                    break
            if random.random() < 0.9:
                N = random.randint(1, len(c.elems) // 2)  # ***
            else:
                N = random.randint(1, len(c.elems))
            collections.extend(c.split(N))
            if c.proxy.issec:
                sections_split += 1
        else:
            # combine
            N = random.randint(2, 10)
            cs = random.sample(collections, N)
            proxies = [c.proxy for c in cs]
            proxy = charm.combine(*proxies)
            c = Collection([], proxy)
            for c_ in cs:
                c.addElems(c_.elems)
            assert hasattr(c.proxy, 'section') and c.proxy.issec
            sections_combined += 1
            collections.append(c)
    print(len(collections), 'collections created, sections_split=', sections_split,
          'sections_combined=', sections_combined)

    if VERBOSE:
        section_sizes = []
        for c in collections:
            if c.proxy.issec is not None:
                section_sizes.append(len(c.elems))
        section_sizes = numpy.array(section_sizes)
        print(len(section_sizes), 'sections, sizes:')
        print('min size=', numpy.min(section_sizes))
        print('median size=', numpy.median(section_sizes))
        print('mean size=', numpy.mean(section_sizes))
        print('max size=', numpy.max(section_sizes))

    for c in collections:
        if c.proxy.issec:
            # this is a section proxy
            sid = c.proxy.section[1]
            c.proxy.recvSecProxy(sid, c.proxy, awaitable=True).get()

    for _ in range(NUM_ITER):
        futures = [Future() for _ in range(len(collections))]
        charm.thisProxy.updateGlobals({'DATA_VERIFY': random.randint(0, 100000)}, awaitable=True).get()
        data = DATA_VERIFY
        for i, c in enumerate(collections):
            sid = None
            if c.proxy.issec:
                sid = c.proxy.section[1]
            c.proxy.recvBcast(data, futures[i], sid)
        for i, f in enumerate(futures):
            result = futures[i].get()
            collections[i].verify(result)

    print('DONE')
    exit()


charm.start(main)
