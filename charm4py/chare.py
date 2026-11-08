from . import wait
from charm4py import ray
import sys
from greenlet import getcurrent
from . import _ckdyn

# A Chare class defined by a user can be used in 4 ways: (1) as a Mainchare, (2) to form Groups,
# (3) to form Arrays, (4) as a singleton chare (`Chare(C, onPE=p)`). Charm4py registers with
# Charm++ (through CkDyn) one class per use, each with its own entry methods, even though the
# definition (body) of the entry methods in Python is the same.
MAINCHARE, GROUP, ARRAY, CHARE = range(4)
CHARM_TYPES = (MAINCHARE, GROUP, ARRAY, CHARE)

# Constants of ckdyn.h (CkDyn API version 1). _ckdyn may export them; the
# values here are the header's and are used when it does not.
def _const(name, value):
    return getattr(_ckdyn, name, value)

# CkDynKind
CKDYN_MAINCHARE = _const('CKDYN_MAINCHARE', 1)
CKDYN_CHARE     = _const('CKDYN_CHARE', 2)
CKDYN_GROUP     = _const('CKDYN_GROUP', 3)
CKDYN_NODEGROUP = _const('CKDYN_NODEGROUP', 4)
CKDYN_ARRAY     = _const('CKDYN_ARRAY', 5)
CKDYN_ARRAYMAP  = _const('CKDYN_ARRAYMAP', 6)
# entry flags
CKDYN_EP_THREADED = _const('CKDYN_EP_THREADED', 1 << 0)
# header flags
CKDYN_MSG_BLOCK       = _const('CKDYN_MSG_BLOCK', 1 << 0)
CKDYN_MSG_BCAST       = _const('CKDYN_MSG_BCAST', 1 << 1)
CKDYN_MSG_BCASTRET    = _const('CKDYN_MSG_BCASTRET', 1 << 2)
CKDYN_MSG_CREATION    = _const('CKDYN_MSG_CREATION', 1 << 3)
CKDYN_MSG_USESATSYNC  = _const('CKDYN_MSG_USESATSYNC', 1 << 4)
CKDYN_MSG_RAY         = _const('CKDYN_MSG_RAY', 1 << 5)
CKDYN_MSG_CONSTRAINED = _const('CKDYN_MSG_CONSTRAINED', 1 << 6)
# binding-private header flags (SPEC-4a 1.1)
PYMSG_REDRESULT = _const('PYMSG_REDRESULT', 1 << 12)
PYMSG_SINGLE    = _const('PYMSG_SINGLE', 1 << 13)
# pup reasons
CKDYN_PUP_MIGRATE    = _const('CKDYN_PUP_MIGRATE', 1)
CKDYN_PUP_CHECKPOINT = _const('CKDYN_PUP_CHECKPOINT', 2)
# element kinds
(CKDYN_ELEM_NONE,
 CKDYN_ELEM_I8, CKDYN_ELEM_I16, CKDYN_ELEM_I32, CKDYN_ELEM_I64,
 CKDYN_ELEM_U8, CKDYN_ELEM_U16, CKDYN_ELEM_U32, CKDYN_ELEM_U64,
 CKDYN_ELEM_F16, CKDYN_ELEM_F32, CKDYN_ELEM_F64,
 CKDYN_ELEM_C64, CKDYN_ELEM_C128,
 CKDYN_ELEM_BOOL, CKDYN_ELEM_BYTE) = range(16)


class Chare(object):

    def __new__(cls, chare_type=None, args=[], onPE=-1, migratable=False):
        # invoked with no arguments when unpickling a chare, and as
        # `Chare(ChareType, args=[...], onPE=p)` to create a singleton chare
        if chare_type is not None:
            # step 4a: always the one-element array (the `migratable=True` form).
            # A CKDYN_CHARE object has a virtual id at the creator and a real id
            # inside the chare, so proxies would compare unequal (channels never
            # connect); the CHARE kind stays registered but is not created here.
            arr = Array(chare_type, ndims=1)
            arr.ckInsert(0, args, onPE, single=True)
            arr.ckDoneInserting()
            proxy = arr[0]
            if hasattr(arr, 'creation_future'):
                proxy.creation_future = arr.creation_future
            return proxy
        return object.__new__(cls)

    def __init__(self):
        if '_numthreads' in self.__dict__:
            return  # already initialized (user __init__ calling super().__init__())
        # stores condition objects which group all elements waiting on same condition string
        self._active_grp_conds = {}
        # linked list of active wait condition objects
        self._cond_next = None
        self._cond_last = self
        self._numthreads = 0
        self._nsuspended = 0  # coroutines of this chare suspended now

    def __flush_wait_queues__(self):
        while True:
            # go through linked list of active wait condition objects
            cond, prev = self._cond_next, self
            if cond is None:
                break
            dequeued = False
            while cond is not None:
                deq, done = cond.check(self)
                dequeued |= deq
                if done:
                    # all elements waiting on this condition have been flushed, remove the condition
                    prev._cond_next = cond._cond_next
                    if cond == self._cond_last:
                        self._cond_last = prev
                    if cond.group:
                        del self._active_grp_conds[cond.cond_str]
                else:
                    prev = cond
                cond = cond._cond_next
            if not dequeued:
                break
            # if I dequeued waiting elements, chare state might have changed as a result of
            # activating them, so I need to continue flushing wait queues

    def __waitEnqueue__(self, cond_template, elem):
        cond_str = cond_template.cond_str
        if cond_template.group and cond_str in self._active_grp_conds:
            self._active_grp_conds[cond_str].enqueue(elem)
        else:
            c = cond_template.createWaitCondition()
            c.enqueue(elem)
            # add to end of linked list of wait condition objects
            self._cond_last._cond_next = c
            self._cond_last = c
            c._cond_next = None
            if cond_template.group:
                self._active_grp_conds[cond_str] = c

    def wait(self, cond_str):
        wait_conditions = self.__class__.__charm_wait_conds__
        if cond_str not in wait_conditions:
            cond_template = wait.ChareStateCond(cond_str, self.__module__)
            wait_conditions[cond_str] = cond_template
        else:
            cond_template = wait_conditions[cond_str]
        if not cond_template.cond_func(self):
            self.__waitEnqueue__(cond_template, (1, getcurrent()))
            charm.threadMgr.pauseThread()

    def contribute(self, data, reducer, callback, section=None):
        charm.contribute(data, reducer, callback, self, section)

    def reduce(self, callback, data=None, reducer=None, section=None):
        assert callable(callback), 'First argument to reduce must be a callback'
        charm.contribute(data, reducer, callback, self, section)

    def allreduce(self, data=None, reducer=None, section=None):
        if section is not None:
            raise NotImplementedError(SECTIONS_ERROR)
        # the future id is shared by every contributor: the reduction number
        # (+1, since 0 means no future) of this contribution
        fid = self.__getRedNo__() + 1
        f = charm.threadMgr.createCollectiveFuture(fid, self, self.thisProxy)
        charm.contribute(data, reducer, f, self, section)
        return f

    def AtSync(self):
        # NOTE this fails if called from a chare that is not in an array (as it should)
        _ckdyn.at_sync(self)

    def migrate(self, toPe):
        if self._nsuspended > 0:
            raise Charm4PyError('Cannot migrate ' + str(self.__class__.__name__) + str(self.thisIndex) +
                                ': it has ' + str(self._nsuspended) + ' suspended coroutine(s)')
        _ckdyn.migrate(self, toPe)

    # called after the chare has migrated to a new PE
    def migrated(self):
        pass

    def setMigratable(self, migratable):
        _ckdyn.set_migratable(self, migratable)
        self.migratable = migratable

    # deposit value of one of the collective futures that was created by this chare
    # (the fid arrives as the callback's reference number, inserted first by _ckdyn)
    def _coll_future_deposit_result(self, fid, result=None):
        charm.threadMgr.depositCollectiveFuture(fid, result, self)

    def __getRedNo__(self):
        return _ckdyn.reduction_number(self)

    def _sectionVote_(self, numsections, section_func, slicing, elems, f, secproxy):  # entry method
        """ charm.split: report the sections (numbered 0 to numsections - 1) this
            member belongs to, as (collection id, index, [section numbers]) gathered
            to the future f """
        if section_func is not None:
            secs = section_func(self)
            if isinstance(secs, int):
                secs = [] if secs < 0 else [secs]
        elif slicing is not None:
            index = self.thisIndex if isinstance(self.thisIndex, tuple) else (self.thisIndex,)
            secs = [0]
            for i, sl in enumerate(slicing):
                step = 1 if sl.step is None else sl.step
                start = 0 if sl.start is None else sl.start
                if index[i] not in range(start, sl.stop, step):
                    secs = []
                    break
        elif elems is not None:
            secs = [n for n, e in enumerate(elems) if self.thisIndex in e]
        else:
            secs = [0]  # every member
        secs = sorted(set(secs))
        assert len(secs) <= numsections, 'Element ' + str(self.thisIndex) + \
                                         ' participates in more sections than were specified'
        proxy = self.thisProxy
        coll_id = proxy.aid if proxy._kind == CKDYN_ARRAY else proxy.gid
        charm.contribute((coll_id, self.thisIndex, secs), Reducer.gather, f, self, secproxy)

    def __addThreadEventSubscriber__(self, target, args):
        self._thread_notify_target = target
        self._thread_notify_data = args

    def __initchannelattrs__(self):
        self.__channels__ = []  # port -> channel._Channel object
        self.__pendingChannels__ = []  # channels that have not finished establishing connections

    def __findPendingChannel__(self, remote, started_locally):
        for i, ch in enumerate(self.__pendingChannels__):
            if ch.locally_initiated == started_locally and ch.remote == remote:
                del self.__pendingChannels__[i]
                return ch
        return None

    def _channelConnect__(self, remote_proxy, remote_port):  # entry method
        if not hasattr(self, '__channels__'):
            self.__initchannelattrs__()
        ch = self.__findPendingChannel__(remote_proxy, True)
        if ch is not None:
            assert not ch.established
            ch.remote_port = remote_port
            if ch.established_fut is not None:
                ch.established_fut.send()
            else:
                ch.setEstablished()
        else:
            from .channel import _Channel
            local_port = len(self.__channels__)
            ch = _Channel(local_port, remote_proxy, False)
            self.__channels__.append(ch)
            self.__pendingChannels__.append(ch)
            ch.remote_port = remote_port

    def _channelRecv__(self, port, seqno, *msg):  # entry method
        ch = self.__channels__[port]
        if len(msg) == 1:
            msg = msg[0]
        ready_fut = ch.wait_ready
        if ready_fut is not None and seqno == ch.recv_seqno:
            ch.data[seqno] = msg
            ch.wait_ready = None
            # signal that channel is ready to receive
            ready_fut.send(ch)
        elif ch.recv_fut is not None and seqno == ch.recv_seqno:
            ch.recv_fut.send(msg)
        else:
            assert seqno not in ch.data, 'Channel buffer is full'
            ch.data[seqno] = msg



method_restrictions = {
    # reserved methods are those that can't be redefined in user subclass
    'reserved': {'__flush_wait_queues__',
                 '__waitEnqueue__', 'wait', 'contribute', 'reduce', 'allreduce',
                 'AtSync', 'migrate', 'setMigratable',
                 '_coll_future_deposit_result', '__getRedNo__', '_sectionVote_',
                 '__addThreadEventSubscriber__',
                 '__initchannelattrs__', '__findPendingChannel__',
                 '_channelConnect__', '_channelRecv__'},

    # these methods of Chare cannot be entry methods. NOTE that any methods starting
    # and ending with '__' are automatically excluded from being entry methods
    'non_entry_method': {'wait', 'contribute', 'reduce', 'allreduce',
                         'AtSync', 'migrated'}
}


def getEntryMethodInfo(cls, method_name):
    func = getattr(cls, method_name)
    argcount = func.__code__.co_argcount - 1  # - 1 to disregard "self" argument
    argnames = tuple(func.__code__.co_varnames[1:argcount + 1])
    assert 'ret' not in argnames, '"ret" keyword for entry method parameters is reserved'
    defaults = func.__defaults__
    if defaults is None:
        defaults = ()
    return argcount, argnames, defaults


def _proxy_methods(cls, charm_type_id, proxyClassName, skip):
    """ The ProxyMethod objects of the generated proxy class of `cls` as
        `charm_type_id`, keyed by method name. Profiling wraps nothing: the C++
        send path is not timed, so printStats has no "send" column. """
    M = {}
    for m in charm.classEntryMethods[charm_type_id][cls]:
        if m.epIdx == -1:
            raise Charm4PyError('Unregistered entry method')
        if m.name in skip:
            continue
        argcount, argnames, defaults = getEntryMethodInfo(m.C, m.name)
        M[m.name] = _ckdyn.make_proxy_method(m.epIdx, argcount, argnames, defaults,
                                             m.name, proxyClassName + '.' + m.name)
    return M


def _base_elemIdx():
    # the read-only elemIdx property of _ckdyn.Proxy
    for klass in _ckdyn.Proxy.__mro__:
        if 'elemIdx' in vars(klass):
            return vars(klass)['elemIdx']
    raise Charm4PyError('_ckdyn.Proxy has no elemIdx')


# ----------------- Mainchare and Proxy -----------------

class Mainchare(object):

    type_id = MAINCHARE

    @classmethod
    def initMember(cls, obj, target):
        obj.thisProxy = charm.proxyClasses[MAINCHARE][obj.__class__]._from_target(target)

    @classmethod
    def __baseEntryMethods__(cls):
        return ['__init__']

    @classmethod
    def __getProxyClass__(C, cls):
        proxyClassName = cls.__name__ + 'Proxy'
        M = _proxy_methods(cls, MAINCHARE, proxyClassName, ('__init__',))
        return type(proxyClassName, (_ckdyn.Proxy,), M)  # create and return proxy class


class DefaultMainchare(Chare):
    def __init__(self, args):
        self.main(args)


# ------------------ Singleton chare and Proxy ------------------

def chare_proxy_elem(proxy, idx):
    # a singleton chare is its own only element (thisProxy[thisIndex] works on it)
    return proxy.__class__._from_target(proxy._target)

def chare_ckNew_gen(C):
    @classmethod
    def chare_ckNew(cls, args, onPE):
        flags, future = 0, None
        if not charm.threadMgr.isMainThread():
            # CkDyn acknowledges the creation to the future
            future = charm.Future()
            flags = CKDYN_MSG_BLOCK | CKDYN_MSG_CREATION
        target = _ckdyn.create_chare(C.idx[CHARE], onPE, tuple(args), flags, future)
        proxy = cls._from_target(target)
        if future is not None:
            proxy.creation_future = future
        return proxy
    return chare_ckNew


class SingletonChare(object):
    """ The singleton chare kind (CKDYN_CHARE): `Chare(C, args, onPE)`. """

    type_id = CHARE

    @classmethod
    def initMember(cls, obj, target):
        obj.thisIndex = (0,)  # as the one-element array that served as singleton before
        obj.thisProxy = charm.proxyClasses[CHARE][obj.__class__]._from_target(target)

    @classmethod
    def __baseEntryMethods__(cls):
        return ['__init__']

    @classmethod
    def __getProxyClass__(C, cls):
        proxyClassName = cls.__name__ + 'ChareProxy'
        M = _proxy_methods(cls, CHARE, proxyClassName, ('__init__',))
        M['__getitem__'] = chare_proxy_elem
        M['ckNew'] = chare_ckNew_gen(cls)
        return type(proxyClassName, (_ckdyn.Proxy,), M)


# ------------------ Group and Proxy  ------------------

def group_proxy_ctor(proxy, gid):
    _ckdyn.Proxy.__init__(proxy, proxy._ckdyn_kind, gid, 0)

def group_proxy_elem(proxy, pe):  # group proxy [] overload method
    if isinstance(pe, slice):
        # a group section (charm.split: called from a coroutine)
        start, stop, step = pe.start, pe.stop, pe.step
        if start is None:
            start = 0
        if stop is None:
            stop = charm.numPes()
        if step is None:
            step = 1
        return charm.split(proxy, 1, elems=[list(range(start, stop, step))])[0]
    # an element of the group (also from a group section proxy)
    proxy_clone = proxy.__class__(proxy.gid)
    proxy_clone._set_elem(pe)
    return proxy_clone

def group_proxy_set_elemIdx(proxy, pe):
    # pool.py assigns elemIdx on a proxy
    if pe == -1:
        proxy._clear_elem()
    else:
        proxy._set_elem(pe)

def update_globals_proxy_method_gen(ep):
    # updateGlobals(global_dict, module_name='__main__') is sent as the flat
    # argument list (name0, value0, name1, value1, ..., module_name)
    def proxy_entry_method(proxy, *args, **kwargs):
        new_args = []
        for varname, var in args[0].items():
            new_args.append(varname)
            new_args.append(var)
        if len(args) >= 2:
            new_args.append(args[1])
        elif 'module_name' in kwargs:
            new_args.append(kwargs['module_name'])
        else:
            new_args.append('__main__')  # default value for 'module_name' parameter
        flags, future = 0, None
        ret = kwargs.get('ret', False)
        if ret or kwargs.get('awaitable', False):
            future = charm.Future()
            flags = CKDYN_MSG_BLOCK
            if proxy.elemIdx == -1:
                flags |= CKDYN_MSG_BCAST
                if ret:
                    flags |= CKDYN_MSG_BCASTRET
        _ckdyn.send(proxy, ep, tuple(new_args), flags, future)
        return future
    proxy_entry_method.ep = ep
    return proxy_entry_method

def group_ckNew_gen(C):
    @classmethod    # make ckNew a class (not instance) method of proxy
    def group_ckNew(cls, args, onPEs):
        flags, future = 0, None
        if not charm.threadMgr.isMainThread() and ArrayMap not in C.mro():
            future = charm.Future()
            flags = CKDYN_MSG_BLOCK | CKDYN_MSG_BCAST  # the constructors contribute to it
        pes = None
        if onPEs is not None:
            pes = [int(pe) for pe in onPEs]  # constrained group
        gid = _ckdyn.create_group(C.idx[GROUP], tuple(args), flags, future, pes)
        proxy = cls(gid)
        if future is not None:
            proxy.creation_future = future
        return proxy
    return group_ckNew

def group_proxy_localbranch(proxy):
    return charm.groups[proxy.gid]

class Group(object):

    type_id = GROUP

    def __new__(cls, C, args=[], onPEs=None):
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError('Only subclasses of Chare can be member of Group')
        if C not in charm.proxyClasses[GROUP]:
            raise Charm4PyError(str(C) + ' not registered for use in Groups')
        return charm.proxyClasses[GROUP][C].ckNew(args, onPEs)

    @classmethod
    def initMember(cls, obj, gid, index=None):
        if index is None:
            index = charm.myPe()
        obj.thisIndex = index
        obj.thisProxy = charm.proxyClasses[GROUP][obj.__class__](gid)
        obj._scookies = defaultdict(int)

    @classmethod
    def __baseEntryMethods__(cls):
        return ['__init__']

    @classmethod
    def __getProxyClass__(C, cls):
        proxyClassName = cls.__name__ + 'GroupProxy'
        M = _proxy_methods(cls, GROUP, proxyClassName, ('__init__',))
        if cls == CharmRemote:
            M['updateGlobals'] = update_globals_proxy_method_gen(M['updateGlobals'].ep)
            M['updateGlobals'].__qualname__ = proxyClassName + '.updateGlobals'
            M['updateGlobals'].__name__ = 'updateGlobals'
            M['exec'] = M['rexec']
        M['__init__'] = group_proxy_ctor
        M['__getitem__'] = group_proxy_elem
        M['elemIdx'] = property(_base_elemIdx().__get__, group_proxy_set_elemIdx)
        M['ckNew'] = group_ckNew_gen(cls)
        M['ckLocalBranch'] = group_proxy_localbranch
        M['_ckdyn_kind'] = CKDYN_ARRAYMAP if ArrayMap in cls.mro() else CKDYN_GROUP
        return type(proxyClassName, (_ckdyn.Proxy,), M)  # create and return proxy class


class ArrayMap(Chare):
    def __init__(self):
        super(ArrayMap, self).__init__()


class NodeGroup(object):
    """ One branch per process (DESIGN.md 5.4 item 1). Not available before step 7. """

    def __new__(cls, C, args=[]):
        raise NotImplementedError('NodeGroup: step 7')


# -------------------- Array and Proxy --------------------

def array_proxy_ctor(proxy, aid, ndims):
    _ckdyn.Proxy.__init__(proxy, CKDYN_ARRAY, aid, ndims)

def array_proxy_elem(proxy, idx):  # array proxy [] overload method
    ndims = proxy.ndims
    isslice = True
    if type(idx) == int:
        idx = (idx,)
        isslice = False
    elif type(idx) == slice:
        idx = (idx,)
    idx = tuple(idx)
    assert len(idx) == ndims, "Dimensions of index " + str(idx) + " don't match array dimensions"
    if isslice and isinstance(idx[0], slice):
        # an array section (charm.split: called from a coroutine)
        for _slice in idx:
            assert _slice.start is not None and _slice.stop is not None, 'Must specify start and stop indexes for array slicing'
        return charm.split(proxy, 1, slicing=idx)[0]
    if proxy.issec and _ckdyn.section_info(proxy.section[1])[2] == -1:
        raise Charm4PyError('element access on a combined section proxy (its elements belong to several arrays)')
    proxy_clone = proxy.__class__(proxy.aid, ndims)
    proxy_clone._set_index(idx)
    return proxy_clone

def array_proxy_set_elemIdx(proxy, idx):
    # pool.py assigns elemIdx on a proxy
    if type(idx) == int:
        idx = (idx,)
    if len(idx) == 0:
        proxy._clear_elem()
    else:
        proxy._set_index(tuple(idx))

def array_ckNew_gen(C):
    @classmethod    # make ckNew a class (not instance) method of proxy
    def array_ckNew(cls, dims=None, ndims=-1, args=[], map=None, useAtSync=False, is_ray=False):
        if type(dims) == int: dims = (dims,)

        if dims is None and ndims == -1:
            raise Charm4PyError('Bounds and number of dimensions for array cannot be empty in ckNew')
        elif dims is not None and ndims != -1 and ndims != len(dims):
            raise Charm4PyError('Number of bounds should match number of dimensions')
        elif dims is None and ndims != -1:  # create an empty array
            dims = (0,) * ndims
        dims = tuple(dims)

        # this is a restriction in Charm++. Charm++ won't tell you unless
        # error checking is enabled, resulting in obscure errors otherwise
        assert charm._myPe == 0, 'Cannot create arrays from PE != 0. Use charm.thisProxy[0].createArray() instead'

        map_gid = -1
        if map is not None:
            map_gid = map.gid

        flags, future = 0, None
        if sum(dims) > 0 and not charm.threadMgr.isMainThread():
            future = charm.Future()
            flags = CKDYN_MSG_BLOCK | CKDYN_MSG_BCAST  # the constructors contribute to it
            if is_ray:
                flags |= CKDYN_MSG_RAY

        create_dims = dims
        if all(d == 0 for d in dims):
            create_dims = None  # empty array, filled with ckInsert
        aid = _ckdyn.create_array(C.idx[ARRAY], create_dims, len(dims), tuple(args), map_gid,
                                  bool(useAtSync), flags, future)
        proxy = cls(aid, len(dims))
        if future is not None:
            proxy.creation_future = future
        return proxy
    return array_ckNew

def array_ckInsert(proxy, index, args=[], onPE=-1, useAtSync=False, single=False, is_ray=False):
    if type(index) == int: index = (index,)
    index = tuple(index)
    assert len(index) == proxy.ndims, 'Invalid index dimensions passed to ckInsert'
    flags, future = 0, None
    if single:
        flags = PYMSG_SINGLE
        if not charm.threadMgr.isMainThread():
            # the constructor's return (None) is sent to the future
            future = charm.Future()
            flags |= CKDYN_MSG_BLOCK
            if is_ray:
                flags |= CKDYN_MSG_RAY
            proxy.creation_future = future
    _ckdyn.array_insert(proxy.aid, index, onPE, tuple(args), flags, future, bool(useAtSync))

def array_proxy_doneInserting(proxy):
    _ckdyn.array_done_inserting(proxy.aid)


class Array(object):

    type_id = ARRAY

    def __new__(cls, C, dims=None, ndims=-1, args=[], map=None, useAtSync=False):
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError('Only subclasses of Chare can be member of Array')
        if C not in charm.proxyClasses[ARRAY]:
            raise Charm4PyError(str(C) + ' not registered for use in Arrays')
        return charm.proxyClasses[ARRAY][C].ckNew(dims, ndims, args, map, useAtSync)

    @classmethod
    def initMember(cls, obj, aid, index, single=False):
        obj.thisIndex = index
        proxy = charm.proxyClasses[ARRAY][obj.__class__](aid, len(obj.thisIndex))
        if single:
            obj.thisProxy = proxy[index]
        else:
            obj.thisProxy = proxy
        obj.migratable = True

    @classmethod
    def __baseEntryMethods__(cls):
        # 'migrated' is run (and profiled) through its EntryMethod object when an
        # element arrives by migration; 'AtSync' is a placeholder that is never
        # sent (Chare.AtSync calls _ckdyn.at_sync)
        return ['__init__', 'migrated', 'AtSync']

    @classmethod
    def __getProxyClass__(C, cls):
        proxyClassName = cls.__name__ + 'ArrayProxy'
        M = _proxy_methods(cls, ARRAY, proxyClassName, ('__init__', 'migrated'))
        M['__init__'] = array_proxy_ctor
        M['__getitem__'] = array_proxy_elem
        M['elemIdx'] = property(_base_elemIdx().__get__, array_proxy_set_elemIdx)
        M['ckNew'] = array_ckNew_gen(cls)
        M['ckInsert'] = array_ckInsert
        M['ckDoneInserting'] = array_proxy_doneInserting
        return type(proxyClassName, (_ckdyn.Proxy,), M)  # create and return proxy class

# ---------------------------------------------------

charm_type_id_to_class = [None] * len(CHARM_TYPES)
for i in CHARM_TYPES:
    if i == MAINCHARE:
        charm_type_id_to_class[i] = Mainchare
    elif i == GROUP:
        charm_type_id_to_class[i] = Group
    elif i == ARRAY:
        charm_type_id_to_class[i] = Array
    elif i == CHARE:
        charm_type_id_to_class[i] = SingletonChare


def charmStarting():
    global charm, Options, Reducer, Charm4PyError, CharmRemote
    from .charm import charm, Charm4PyError, CharmRemote
    Options = charm.options
    Reducer = charm.reducers
