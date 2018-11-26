from . import wait
import sys
if sys.version_info[0] < 3:
    from thread import get_ident
else:
    from threading import get_ident


# A Chare class defined by a user can be used in 3 ways: (1) as a Mainchare, (2) to form Groups,
# (3) to form Arrays. To achieve this, charm4py can register with the Charm++ library up to 3
# different types for the given class (a Mainchare, a Group and an Array), and each type will
# register its own entry methods, even though the definition (body) of the entry methods in Python is the same.
MAINCHARE, GROUP, ARRAY = range(3)
CHARM_TYPES = (MAINCHARE, GROUP, ARRAY)

# Constants to detect type of contributors for reduction. Order should match enum extContributorType
(CONTRIBUTOR_TYPE_ARRAY,
 CONTRIBUTOR_TYPE_GROUP,
 CONTRIBUTOR_TYPE_NODEGROUP) = range(3)


class Chare(object):

    def __new__(cls, chare_type=None, args=[], onPE=-1):
        # this method is only invoked when unpickling a chare (invoked with no arguments), or
        # when creating a singleton chare with `Chare(ChareType, args=[...], onPE=p)`
        if chare_type is not None:
            arr = Array(chare_type, ndims=1)
            arr.ckInsert(0, args, onPE)
            return arr[0]
        return object.__new__(cls)

    def __init__(self):
        if hasattr(self, '_chare_initialized'):
            return
        # messages to this chare from chares in the same PE are stored here without copying
        # or pickling. _local is a fixed size array that implements a mem pool, where msgs
        # can be in non-consecutive positions, and the indexes of free slots are stored
        # as a linked list inside _local, with _local_free_head being the index of the
        # first free slot, _local[_local_free_head] is the index of next free slot and so on
        self._local = [i for i in range(1, Options.LOCAL_MSG_BUF_SIZE+1)]
        self._local[-1] = None
        self._local_free_head = 0
        # stores condition objects which group all elements waiting on same condition string
        self._active_grp_conds = {}
        # linked list of active wait condition objects
        self._cond_next = None
        self._cond_last = self
        self._chare_initialized = True

    def __addLocal__(self, msg):
        if self._local_free_head is None:
            raise Charm4PyError("Local msg buffer full. Increase LOCAL_MSG_BUF_SIZE")
        h = self._local_free_head
        self._local_free_head = self._local[self._local_free_head]
        self._local[h] = msg
        return h

    def __removeLocal__(self, tag):
        msg = self._local[tag]
        self._local[tag] = self._local_free_head
        self._local_free_head = tag
        return msg

    def __flush_wait_queues__(self):
        while True:
            # go through linked list of active wait condition objects
            cond, prev = self._cond_next, self
            if cond is None: break
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
            if not dequeued: break
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
        if cond_str not in charm.wait_conditions:
            cond_template = wait.ChareStateCond(cond_str)
            charm.wait_conditions[cond_str] = cond_template
        else:
            cond_template = charm.wait_conditions[cond_str]
        if not cond_template.cond_func(self):
            self.__waitEnqueue__(cond_template, (1, get_ident()))
            charm.threadMgr.pauseThread()

    def contribute(self, data, reducer_type, target):
        charm.contribute(data, reducer_type, target, self)

    def AtSync(self):
        # NOTE this will fail if called from a chare that is not in an array (as it should be)
        charm.CkArraySend(self.thisProxy.aid, self.thisIndex, self.thisProxy.AtSync.ep, (b'', []))

    def migrate(self, toPe):
        # print("[charm4py] Calling migrate, aid: ", self.thisProxy.aid, "ndims",
        #       self.thisProxy.ndims, "index: ", self.thisIndex, "toPe", toPe)
        charm.lib.CkMigrate(self.thisProxy.aid, self.thisIndex, toPe)

    # deposit value of one of the futures that was created on this chare
    def _future_deposit_result(self, fid, result=None):
        charm.threadMgr.depositFuture(fid, result)


method_restrictions = {
    # reserved methods are those that can't be redefined in user subclass
    'reserved': {'__addLocal__', '__removeLocal__', '__flush_wait_queues__',
                 '__waitEnqueue__', 'wait', 'contribute', 'AtSync',
                 'migrate', '_future_deposit_result'},

    # these methods of Chare cannot be entry methods. NOTE that any methods starting
    # and ending with '__' are automatically excluded from being entry methods
    'non_entry_method': {'wait', 'contribute', 'AtSync', 'migrate'}
}

# ----------------- Mainchare and Proxy -----------------

def mainchare_proxy_ctor(proxy, cid):
    proxy.cid = cid

def mainchare_proxy__getstate__(proxy):
    return proxy.cid

def mainchare_proxy__setstate__(proxy, state):
    proxy.cid = state

def mainchare_proxy_method_gen(ep):  # decorator, generates proxy entry methods
    def proxy_entry_method(proxy, *args, **kwargs):
        header = {}
        blockFuture = None
        cid = proxy.cid  # chare ID
        if 'ret' in kwargs and kwargs['ret']:
            header[b'block'] = blockFuture = charm.createFuture()
        destObj = None
        if Options.LOCAL_MSG_OPTIM and (cid in charm.chares) and (len(args) > 0):
            destObj = charm.chares[cid]
        msg = charm.packMsg(destObj, args, header)
        charm.CkChareSend(cid, ep, msg)
        return blockFuture
    proxy_entry_method.ep = ep
    return proxy_entry_method

def mainchare_proxy_contribute(proxy, contributeInfo):
    charm.CkContributeToChare(contributeInfo, proxy.cid)


class Mainchare(object):

    type_id = MAINCHARE

    @classmethod
    def initMember(cls, obj, cid):
        obj.thisProxy = charm.proxyClasses[MAINCHARE][obj.__class__](cid)

    @classmethod
    def __baseEntryMethods__(cls):
        return ["__init__"]

    @classmethod
    def __getProxyClass__(C, cls):
        # print("Creating mainchare proxy class for class " + cls.__name__)
        M = dict()  # proxy methods
        for m in charm.classEntryMethods[MAINCHARE][cls]:
            if m.epIdx == -1:
                raise Charm4PyError("Unregistered entry method")
            if Options.PROFILING:
                M[m.name] = profile_send_function(mainchare_proxy_method_gen(m.epIdx))
            else:
                M[m.name] = mainchare_proxy_method_gen(m.epIdx)
        M["__init__"] = mainchare_proxy_ctor
        M["ckContribute"] = mainchare_proxy_contribute  # function called when target proxy is Mainchare
        M["__getstate__"] = mainchare_proxy__getstate__
        M["__setstate__"] = mainchare_proxy__setstate__
        return type(cls.__name__ + 'Proxy', (), M)  # create and return proxy class


class DefaultMainchare(Chare):
    def __init__(self, args):
        self.main(args)


# ------------------ Group and Proxy  ------------------

def group_proxy_ctor(proxy, gid):
    proxy.gid = gid
    proxy.elemIdx = -1  # entry method calls will be to elemIdx PE (broadcast if -1)

def group_proxy__getstate__(proxy):
    return (proxy.gid, proxy.elemIdx)

def group_proxy__setstate__(proxy, state):
    proxy.gid, proxy.elemIdx = state

def group_proxy_elem(proxy, pe):  # group proxy [] overload method
    proxy_clone = proxy.__class__(proxy.gid)
    proxy_clone.elemIdx = pe
    return proxy_clone

def group_proxy_method_gen(ep):  # decorator, generates proxy entry methods
    def proxy_entry_method(proxy, *args, **kwargs):
        header = {}
        blockFuture = None
        elemIdx = proxy.elemIdx
        if 'ret' in kwargs and kwargs['ret']:
            header[b'block'] = blockFuture = charm.createFuture()
            if elemIdx == -1:
                header[b'bcast'] = True
        destObj = None
        if Options.LOCAL_MSG_OPTIM and (elemIdx == charm._myPe) and (len(args) > 0):
            destObj = charm.groups[proxy.gid]
        msg = charm.packMsg(destObj, args, header)
        charm.CkGroupSend(proxy.gid, elemIdx, ep, msg)
        return blockFuture
    proxy_entry_method.ep = ep
    return proxy_entry_method

def group_ckNew_gen(C, epIdx):
    @classmethod    # make ckNew a class (not instance) method of proxy
    def group_ckNew(cls, args):
        # print("GROUP calling ckNew for class " + C.__name__ + " cIdx=", C.idx[GROUP], "epIdx=", epIdx)
        header, creation_future = {}, None
        if get_ident() != charm.threadMgr.main_thread_id and ArrayMap not in C.mro():
            creation_future = charm.createFuture()
            header[b'block'] = creation_future
        msg = charm.packMsg(None, args, header)
        gid = charm.lib.CkCreateGroup(C.idx[GROUP], epIdx, msg)
        proxy = charm.groups[gid].thisProxy
        if creation_future is not None:
            proxy.creation_future = creation_future
        return proxy
    return group_ckNew

def group_proxy_contribute(proxy, contributeInfo):
    charm.CkContributeToGroup(contributeInfo, proxy.gid, proxy.elemIdx)


class Group(object):

    type_id = GROUP

    def __new__(cls, C, args=[]):
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError("Only subclasses of Chare can be member of Group")
        if C not in charm.proxyClasses[GROUP]:
            raise Charm4PyError(str(C) + ' not registered for use in Groups')
        return charm.proxyClasses[GROUP][C].ckNew(args)

    @classmethod
    def initMember(cls, obj, gid):
        obj.thisIndex = charm.myPe()
        obj.thisProxy = charm.proxyClasses[GROUP][obj.__class__](gid)
        obj._contributeInfo = charm.lib.initContributeInfo(gid, obj.thisIndex, CONTRIBUTOR_TYPE_GROUP)

    @classmethod
    def __baseEntryMethods__(cls):
        return ["__init__"]

    @classmethod
    def __getProxyClass__(C, cls):
        # print("Creating group proxy class for class " + cls.__name__)
        M = dict()  # proxy methods
        entryMethods = charm.classEntryMethods[GROUP][cls]
        for m in entryMethods:
            if m.epIdx == -1:
                raise Charm4PyError("Unregistered entry method")
            if Options.PROFILING:
                M[m.name] = profile_send_function(group_proxy_method_gen(m.epIdx))
            else:
                M[m.name] = group_proxy_method_gen(m.epIdx)
        M["__init__"] = group_proxy_ctor
        M["__getitem__"] = group_proxy_elem
        M["ckNew"] = group_ckNew_gen(cls, entryMethods[0].epIdx)
        M["ckContribute"] = group_proxy_contribute  # function called when target proxy is Group
        M["__getstate__"] = group_proxy__getstate__
        M["__setstate__"] = group_proxy__setstate__
        return type(cls.__name__ + 'GroupProxy', (), M)  # create and return proxy class


class ArrayMap(Chare):
    def __init__(self):
        super(ArrayMap, self).__init__()


# -------------------- Array and Proxy --------------------

def array_proxy_ctor(proxy, aid, ndims):
    proxy.aid = aid
    proxy.ndims = ndims
    proxy.elemIdx = ()  # entry method calls will be to elemIdx array element (broadcast if empty tuple)

def array_proxy__getstate__(proxy):
    return (proxy.aid, proxy.ndims, proxy.elemIdx)

def array_proxy__setstate__(proxy, state):
    proxy.aid, proxy.ndims, proxy.elemIdx = state

def array_proxy_elem(proxy, idx):  # array proxy [] overload method
    proxy_clone = proxy.__class__(proxy.aid, proxy.ndims)
    if type(idx) == int: idx = (idx,)
    if len(idx) != proxy_clone.ndims:
        raise Charm4PyError("Dimensions of index " + str(idx) + " don't match array dimensions")
    proxy_clone.elemIdx = tuple(idx)
    return proxy_clone

def array_proxy_method_gen(ep):  # decorator, generates proxy entry methods
    def proxy_entry_method(proxy, *args, **kwargs):
        header = {}
        blockFuture = None
        elemIdx = proxy.elemIdx
        if 'ret' in kwargs and kwargs['ret']:
            header[b'block'] = blockFuture = charm.createFuture()
            if elemIdx == ():
                header[b'bcast'] = True
        destObj = None
        if Options.LOCAL_MSG_OPTIM and (len(args) > 0):
            array = charm.arrays[proxy.aid]
            if elemIdx in array:
                destObj = array[elemIdx]
        msg = charm.packMsg(destObj, args, header)
        charm.CkArraySend(proxy.aid, elemIdx, ep, msg)
        return blockFuture
    proxy_entry_method.ep = ep
    return proxy_entry_method

def array_ckNew_gen(C, epIdx):
    @classmethod    # make ckNew a class (not instance) method of proxy
    def array_ckNew(cls, dims=None, ndims=-1, args=[], map=None):
        # if charm.myPe() == 0: print("calling array ckNew for class " + C.__name__ + " cIdx=" + str(C.idx[ARRAY]))
        if type(dims) == int: dims = (dims,)

        if dims is None and ndims == -1:
            raise Charm4PyError("Bounds and number of dimensions for array cannot be empty in ckNew")
        elif dims is not None and ndims != -1 and ndims != len(dims):
            raise Charm4PyError("Number of bounds should match number of dimensions")
        elif dims is None and ndims != -1:  # create an empty array
            dims = (0,) * ndims

        map_gid = -1
        if map is not None:
            map_gid = map.gid

        header, creation_future = {}, None
        if get_ident() != charm.threadMgr.main_thread_id:
            creation_future = charm.createFuture()
            header[b'block'] = creation_future

        msg = charm.packMsg(None, args, header)
        aid = charm.lib.CkCreateArray(C.idx[ARRAY], dims, epIdx, msg, map_gid)
        proxy = cls(aid, len(dims))
        if creation_future is not None:
            proxy.creation_future = creation_future
        return proxy
    return array_ckNew

def array_ckInsert_gen(epIdx):
    def array_ckInsert(proxy, index, args=[], onPE=-1):
        if type(index) == int: index = (index,)
        assert len(index) == proxy.ndims, "Invalid index dimensions passed to ckInsert"
        msg = charm.packMsg(None, args, {})
        charm.lib.CkInsert(proxy.aid, index, epIdx, onPE, msg)
    return array_ckInsert

def array_proxy_contribute(proxy, contributeInfo):
    charm.CkContributeToArray(contributeInfo, proxy.aid, proxy.elemIdx)

def array_proxy_doneInserting(proxy):
    charm.lib.CkDoneInserting(proxy.aid)


class Array(object):

    type_id = ARRAY

    def __new__(cls, C, dims=None, ndims=-1, args=[], map=None):
        if (not hasattr(C, 'mro')) or (Chare not in C.mro()):
            raise Charm4PyError("Only subclasses of Chare can be member of Array")
        if C not in charm.proxyClasses[ARRAY]:
            raise Charm4PyError(str(C) + ' not registered for use in Arrays')
        return charm.proxyClasses[ARRAY][C].ckNew(dims, ndims, args, map)

    @classmethod
    def initMember(cls, obj, aid, index):
        obj.thisIndex = index
        obj.thisProxy = charm.proxyClasses[ARRAY][obj.__class__](aid, len(obj.thisIndex))
        # NOTE currently only used at Python level. proxy object in charm runtime currently has this set to true
        obj.usesAtSync = False
        obj._contributeInfo = charm.lib.initContributeInfo(aid, obj.thisIndex, CONTRIBUTOR_TYPE_ARRAY)

    @classmethod
    def __baseEntryMethods__(cls):
        # 2nd __init__ used to register migration constructor
        return ["__init__", "__init__", "AtSync"]

    @classmethod
    def __getProxyClass__(C, cls):
        # print("Creating array proxy class for class " + cls.__name__)
        M = dict()  # proxy methods
        entryMethods = charm.classEntryMethods[ARRAY][cls]
        for m in entryMethods:
            if m.epIdx == -1:
                raise Charm4PyError("Unregistered entry method")
            if Options.PROFILING:
                M[m.name] = profile_send_function(array_proxy_method_gen(m.epIdx))
            else:
                M[m.name] = array_proxy_method_gen(m.epIdx)
        M["__init__"] = array_proxy_ctor
        M["__getitem__"] = array_proxy_elem
        M["ckNew"] = array_ckNew_gen(cls, entryMethods[0].epIdx)
        M["ckInsert"] = array_ckInsert_gen(entryMethods[0].epIdx)
        M["ckContribute"] = array_proxy_contribute  # function called when target proxy is Array
        M["ckDoneInserting"] = array_proxy_doneInserting
        M["__getstate__"] = array_proxy__getstate__
        M["__setstate__"] = array_proxy__setstate__
        return type(cls.__name__ + 'ArrayProxy', (), M)  # create and return proxy class


# ---------------------------------------------------

charm_type_id_to_class = [None] * len(CHARM_TYPES)
for i in CHARM_TYPES:
    if i == MAINCHARE:
        charm_type_id_to_class[i] = Mainchare
    elif i == GROUP:
        charm_type_id_to_class[i] = Group
    elif i == ARRAY:
        charm_type_id_to_class[i] = Array


def charmStarting():
    from .charm import charm, Options, Charm4PyError, profile_send_function
    globals()['charm'] = charm
    globals()['Reducer'] = charm.reducers
    globals()['Options'] = Options
    globals()['Charm4PyError'] = Charm4PyError
    globals()['profile_send_function'] = profile_send_function
