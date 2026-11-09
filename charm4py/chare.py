from . import wait
from charm4py import ray
import sys
from greenlet import getcurrent
from collections import defaultdict

# A Chare class defined by a user can be used in 3 ways: (1) as a Mainchare, (2) to form Groups,
# (3) to form Arrays. To achieve this, Charm4py can register with the Charm++ library up to 3
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
            arr.ckInsert(0, args, onPE, single=True)
            arr.ckDoneInserting()
            proxy = arr[0]
            if hasattr(arr, 'creation_future'):
                proxy.creation_future = arr.creation_future
            return proxy
        return object.__new__(cls)

    def __init__(self):
        if hasattr(self, '_local'):
            return
        # messages to this chare from chares in the same PE are stored here without copying
        # or pickling. _local is a fixed size array that implements a mem pool, where msgs
        # can be in non-consecutive positions, and the indexes of free slots are stored
        # as a linked list inside _local, with _local_free_head being the index of the
        # first free slot, _local[_local_free_head] is the index of next free slot and so on
        self._local = [i for i in range(1, Options.local_msg_buf_size + 1)]
        self._local[-1] = None
        self._local_free_head = 0
        # stores condition objects which group all elements waiting on same condition string
        self._active_grp_conds = {}
        # linked list of active wait condition objects
        self._cond_next = None
        self._cond_last = self
        self._numthreads = 0

    def __addLocal__(self, msg):
        if self._local_free_head is None:
            raise Charm4PyError('Local msg buffer full. Increase LOCAL_MSG_BUF_SIZE')
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
        if section is None:
            # The CMK_REFNUM_TYPE type inside Charm++ CkCallbacks that we use
            # to carry future IDs (aka fid) is usually unsigned short, so we
            # don't go over the max value for that type. Also, we can't have
            # fid==0 because that means no fid
            fid = self.__getRedNo__() % 65535 + 1
            proxy = self.thisProxy
        else:
            # Note that this currently works because section reductions are
            # entirely implemented in sections.py. Some adjustments will be
            # needed when the implementation moves to Charm++ (fid/redno will
            # have to start at 1, and the fid carried in Charm++ CkCallbacks
            # can only contain the 'redno' part
            sid = section.section[1]
            assert self._scookies[sid] < 65535
            redno = self._scookies[sid] % 65535
            # An object can participate in multiple sections. fids need to be
            # unique across sections, and not conflict with non-section fids
            fid = (redno, sid[0], sid[1])
            proxy = section
        f = charm.threadMgr.createCollectiveFuture(fid, self, proxy)
        charm.contribute(data, reducer, f, self, section)
        return f

    def AtSync(self):
        # NOTE this will fail if called from a chare that is not in an array (as it should be)
        charm.CkArraySend(self.thisProxy.aid, self.thisIndex, self.thisProxy.AtSync.ep, (b'', []))

    def migrate(self, toPe):
        charm.lib.CkMigrate(self.thisProxy.aid, self.thisIndex, toPe)

    # called after the chare has migrated to a new PE
    def migrated(self):
        pass

    def setMigratable(self, migratable):
        charm.lib.setMigratable(self.thisProxy.aid, self.thisIndex, migratable)
        self.migratable = migratable

    # deposit value of one of the collective futures that was created by this chare
    def _coll_future_deposit_result(self, fid, result=None):
        charm.threadMgr.depositCollectiveFuture(fid, result, self)

    def __getRedNo__(self):
        proxy = self.thisProxy
        if hasattr(proxy, 'aid'):
            return charm.lib.getArrayElementRedNo(proxy.aid, self.thisIndex)
        else:
            return charm.lib.getGroupRedNo(proxy.gid)

    def __addThreadEventSubscriber__(self, target, args):
        self._thread_notify_target = target
        self._thread_notify_data = args

    def _getSectionLocations_(self, sid0, numsections, member_func, slicing, section_elems, f, proxy):
        # list of sections in which this element participates (sections
        # numbered from 0 to numsections - 1)
        sections = []
        if member_func is not None:
            sections = member_func(self)
            if isinstance(sections, int):
                if sections < 0:
                    sections = []  # I don't belong to any section
                else:
                    sections = [sections]
        elif slicing is not None:
            insection = True
            for i, idx_i in enumerate(self.thisIndex):
                sl = slicing[i]
                step = 1
                if sl.step is not None:
                    step = sl.step
                if idx_i not in range(sl.start, sl.stop, step):
                    insection = False
                    break
            if insection:
                sections.append(0)  # when slicing there is only one section (0)
        else:
            for sec_num, elems in enumerate(section_elems):
                if self.thisIndex in elems:
                    sections.append(sec_num)
        assert len(sections) <= numsections, 'Element ' + str(self.thisIndex) + \
                                             ' participates in more sections than were specified'
        if len(sections) > 0 and not hasattr(self, '_scookies'):
            # chares that participate in sections need this dict to store their
            # reduction numbers for each section
            self._scookies = defaultdict(int)
        result = [set() for _ in range(numsections)]
        sid_pe, sid_cnt_start = sid0
        mype = charm.myPe()
        for sec_num in sections:
            sid = (sid_pe, sid_cnt_start + sec_num)
            result[sec_num].add(mype)
            # We don't use a set for local_elems because section creation is not frequent,
            # and the most frequent use of local_elems will be to iterate through its elements
            # (a list is a bit faster than set for this)
            local_elems = charm.sectionMgr.sections[sid].local_elems
            if self not in local_elems:
                local_elems.append(self)
        # send result to future via a reduction
        charm.contribute(result, Reducer._sectionloc, f, self, proxy)

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
    'reserved': {'__addLocal__', '__removeLocal__', '__flush_wait_queues__',
                 '__waitEnqueue__', 'wait', 'contribute', 'reduce', 'allreduce',
                 'AtSync', 'migrate', 'setMigratable',
                 '_coll_future_deposit_result', '__getRedNo__',
                 '__addThreadEventSubscriber__', '_getSectionLocations_',
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

# ----------------- Mainchare and Proxy -----------------

def mainchare_proxy_ctor(proxy, cid):
    proxy.cid = cid

def mainchare_proxy__getstate__(proxy):
    return proxy.cid

def mainchare_proxy__setstate__(proxy, state):
    proxy.cid = state

def mainchare_proxy__eq__(proxy, other):
    if isinstance(other, proxy.__class__):
        return proxy.cid == other.cid
    else:
        return False

def mainchare_proxy__hash__(proxy):
    return hash(proxy.cid)

def mainchare_proxy_method_gen(ep, argcount, argnames, defaults):  # decorator, generates proxy entry methods
    def proxy_entry_method(proxy, *args, **kwargs):
        num_args = len(args)
        if num_args < argcount and len(kwargs) > 0:
            args = list(args)
            for i in range(num_args, argcount):
                argname = argnames[i]
                # first look for argument in kwargs
                if argname in kwargs:
                    args.append(kwargs[argname])
                else:
                    # if not there, see if there is a default value
                    def_idx = i - argcount + len(defaults)
                    assert def_idx >= 0, 'Value not found for parameter \'' + argname + '\' of entry method'
                    args.append(defaults[def_idx])

        header = {}
        blockFuture = None
        cid = proxy.cid  # chare ID
        if ('ret' in kwargs and kwargs['ret']) or ('awaitable' in kwargs and kwargs['awaitable']):
            header[b'block'] = blockFuture = charm.Future()
        destObj = None
        if Options.local_msg_optim and (cid in charm.chares) and (len(args) > 0):
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
        return ['__init__']

    @classmethod
    def __getProxyClass__(C, cls):
        # print("Creating mainchare proxy class for class " + cls.__name__)
        proxyClassName = cls.__name__ + 'Proxy'
        M = dict()  # proxy methods
        for m in charm.classEntryMethods[MAINCHARE][cls]:
            if m.epIdx == -1:
                raise Charm4PyError('Unregistered entry method')
            if m.name == '__init__':
                continue
            argcount, argnames, defaults = getEntryMethodInfo(m.C, m.name)
            if Options.profiling:
                f = profile_send_function(mainchare_proxy_method_gen(m.epIdx, argcount, argnames, defaults))
            else:
                f = mainchare_proxy_method_gen(m.epIdx, argcount, argnames, defaults)
            f.__qualname__ = proxyClassName + '.' + m.name
            f.__name__ = m.name
            M[m.name] = f
        M['__init__'] = mainchare_proxy_ctor
        M['ckContribute'] = mainchare_proxy_contribute  # function called when target proxy is Mainchare
        M['__getstate__'] = mainchare_proxy__getstate__
        M['__setstate__'] = mainchare_proxy__setstate__
        M['__eq__'] = mainchare_proxy__eq__
        M['__hash__'] = mainchare_proxy__hash__
        return type(proxyClassName, (), M)  # create and return proxy class


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

def group_proxy__eq__(proxy, other):
    if proxy.issec:
        if hasattr(other, 'issec'):
            return proxy.section == other.section
        else:
            return False
    elif isinstance(other, proxy.__class__):
        return proxy.gid == other.gid and proxy.elemIdx == other.elemIdx
    else:
        return False

def group_proxy__hash__(proxy):
    if proxy.issec:
        return hash(proxy.section)
    else:
        return hash((proxy.gid, proxy.elemIdx))

def group_getsecproxy(proxy, sinfo):
    if proxy.issec:
        secproxy = proxy.__class__(proxy.gid)
    else:
        secproxy = proxy.__secproxyclass__(proxy.gid)
    secproxy.section = sinfo
    return secproxy

def groupsecproxy__getstate__(proxy):
    return (proxy.gid, proxy.elemIdx, proxy.section)

def groupsecproxy__setstate__(proxy, state):
    proxy.gid, proxy.elemIdx, proxy.section = state

def group_proxy_elem(proxy, pe):  # group proxy [] overload method
    if not isinstance(pe, slice):
        proxy_clone = proxy.__class__(proxy.gid)
        proxy_clone.elemIdx = pe
        return proxy_clone
    else:
        start, stop, step = pe.start, pe.stop, pe.step
        if start is None:
            start = 0
        if stop is None:
            stop = charm.numPes()
        if step is None:
            step = 1
        return charm.split(proxy, 1, elems=[list(range(start, stop, step))])[0]

def group_proxy_method_gen(ep, argcount, argnames, defaults):  # decorator, generates proxy entry methods
    def proxy_entry_method(proxy, *args, **kwargs):
        num_args = len(args)
        if num_args < argcount and len(kwargs) > 0:
            args = list(args)
            for i in range(num_args, argcount):
                argname = argnames[i]
                # first look for argument in kwargs
                if argname in kwargs:
                    args.append(kwargs[argname])
                else:
                    # if not there, see if there is a default value
                    def_idx = i - argcount + len(defaults)
                    assert def_idx >= 0, 'Value not found for parameter \'' + argname + '\' of entry method'
                    args.append(defaults[def_idx])

        header = {}
        blockFuture = None
        elemIdx = proxy.elemIdx
        if 'ret' in kwargs and kwargs['ret']:
            header[b'block'] = blockFuture = charm.Future()
            if elemIdx == -1:
                header[b'bcast'] = header[b'bcastret'] = True
        elif 'awaitable' in kwargs and kwargs['awaitable']:
            header[b'block'] = blockFuture = charm.Future()
            if elemIdx == -1:
                header[b'bcast'] = True
        if not proxy.issec or elemIdx != -1:
            destObj = None
            gid = proxy.gid
            if Options.local_msg_optim and (elemIdx == charm._myPe) and (len(args) > 0):
                destObj = charm.groups[gid]
            msg = charm.packMsg(destObj, args, header)
            charm.CkGroupSend(gid, elemIdx, ep, msg)
        else:
            root, sid = proxy.section
            header[b'sid'] = sid
            if Options.local_msg_optim and root == charm._myPe:
                charm.sectionMgr.thisProxy[root].sendToSectionLocal(sid, ep, header, *args)
            else:
                charm.sectionMgr.thisProxy[root].sendToSection(sid, ep, header, *args)
        return blockFuture
    proxy_entry_method.ep = ep
    return proxy_entry_method

def update_globals_proxy_method_gen(ep):
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
        args = new_args
        header = {}
        blockFuture = None
        elemIdx = proxy.elemIdx
        if 'ret' in kwargs and kwargs['ret']:
            header[b'block'] = blockFuture = charm.Future()
            if elemIdx == -1:
                header[b'bcast'] = header[b'bcastret'] = True
        elif 'awaitable' in kwargs and kwargs['awaitable']:
            header[b'block'] = blockFuture = charm.Future()
            if elemIdx == -1:
                header[b'bcast'] = True
        if not proxy.issec or elemIdx != -1:
            destObj = None
            gid = proxy.gid
            if Options.local_msg_optim and (elemIdx == charm._myPe) and (len(args) > 0):
                destObj = charm.groups[gid]
            msg = charm.packMsg(destObj, args, header)
            charm.CkGroupSend(gid, elemIdx, ep, msg)
        else:
            root, sid = proxy.section
            header[b'sid'] = sid
            if Options.local_msg_optim and root == charm._myPe:
                charm.sectionMgr.thisProxy[root].sendToSectionLocal(sid, ep, header, *args)
            else:
                charm.sectionMgr.thisProxy[root].sendToSection(sid, ep, header, *args)
        return blockFuture
    proxy_entry_method.ep = ep
    return proxy_entry_method

def group_ckNew_gen(C, epIdx):
    @classmethod    # make ckNew a class (not instance) method of proxy
    def group_ckNew(cls, args, onPEs):
        # print("GROUP calling ckNew for class " + C.__name__ + " cIdx=", C.idx[GROUP], "epIdx=", epIdx)
        header = {}
        creation_future = None
        if not charm.threadMgr.isMainThread() and ArrayMap not in C.mro():
            creation_future = charm.Future()
            header[b'block'] = creation_future
            header[b'bcast'] = True
            header[b'creation'] = True
        if onPEs is None:
            msg = charm.packMsg(None, args, header)
            gid = charm.lib.CkCreateGroup(C.idx[GROUP], epIdx, msg)
            proxy = cls(gid)
        else:
            # send empty msg for Charm++ group creation (on every PE)
            msg = charm.packMsg(None, [], {b'constrained': True})
            gid = charm.lib.CkCreateGroup(C.idx[GROUP], epIdx, msg)
            proxy = cls(gid)
            # real msg goes only to section elements
            proxy = charm.split(proxy, 1, elems=[onPEs], cons=[-1, epIdx, header, args])[0]
        if creation_future is not None:
            proxy.creation_future = creation_future
        return proxy
    return group_ckNew

def group_proxy_contribute(proxy, contributeInfo):
    charm.CkContributeToGroup(contributeInfo, proxy.gid, proxy.elemIdx)

def groupsecproxy_contribute(proxy, contributeInfo):
    charm.CkContributeToSection(contributeInfo, proxy.section[1], proxy.section[0])

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
    def initMember(cls, obj, gid):
        obj.thisIndex = charm.myPe()
        obj.thisProxy = charm.proxyClasses[GROUP][obj.__class__](gid)
        obj._contributeInfo = charm.lib.initContributeInfo(gid, obj.thisIndex, CONTRIBUTOR_TYPE_GROUP)
        obj._scookies = defaultdict(int)

    @classmethod
    def __baseEntryMethods__(cls):
        return ['__init__']

    @classmethod
    def __getProxyClass__(C, cls, sectionProxy=False):
        # print("Creating group proxy class for class " + cls.__name__)
        if not sectionProxy:
            proxyClassName = cls.__name__ + 'GroupProxy'
        else:
            proxyClassName = cls.__name__ + 'GroupSecProxy'
        M = dict()  # proxy methods
        entryMethods = charm.classEntryMethods[GROUP][cls]
        for m in entryMethods:
            if m.epIdx == -1:
                raise Charm4PyError('Unregistered entry method')
            if m.name == '__init__':
                continue
            if m.name == 'updateGlobals' and cls == CharmRemote:
                if Options.profiling:
                    f = profile_send_function(update_globals_proxy_method_gen(m.epIdx))
                else:
                    f = update_globals_proxy_method_gen(m.epIdx)
            else:
                argcount, argnames, defaults = getEntryMethodInfo(m.C, m.name)
                if Options.profiling:
                    f = profile_send_function(group_proxy_method_gen(m.epIdx, argcount, argnames, defaults))
                else:
                    f = group_proxy_method_gen(m.epIdx, argcount, argnames, defaults)
            f.__qualname__ = proxyClassName + '.' + m.name
            f.__name__ = m.name
            M[m.name] = f
        if cls == CharmRemote and sys.version_info >= (3, 0, 0):
            # TODO remove this and change rexec to exec when Python 2 support is dropped
            M['exec'] = M['rexec']
        M['__init__'] = group_proxy_ctor
        M['__getitem__'] = group_proxy_elem
        M['__eq__'] = group_proxy__eq__
        M['__hash__'] = group_proxy__hash__
        M['ckNew'] = group_ckNew_gen(cls, entryMethods[0].epIdx)
        M['ckLocalBranch'] = group_proxy_localbranch
        M['__getsecproxy__'] = group_getsecproxy
        if not sectionProxy:
            M['ckContribute'] = group_proxy_contribute  # function called when target proxy is Group
            M['__getstate__'] = group_proxy__getstate__
            M['__setstate__'] = group_proxy__setstate__
        else:
            M['ckContribute'] = groupsecproxy_contribute  # function called when target proxy is Group
            M['__getstate__'] = groupsecproxy__getstate__
            M['__setstate__'] = groupsecproxy__setstate__
        proxyCls = type(proxyClassName, (), M)  # create and return proxy class
        proxyCls.issec = sectionProxy
        return proxyCls


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

def array_proxy__eq__(proxy, other):
    if proxy.issec:
        if hasattr(other, 'issec'):
            return proxy.section == other.section
        else:
            return False
    elif isinstance(other, proxy.__class__):
        return proxy.aid == other.aid and proxy.elemIdx == other.elemIdx
    else:
        return False

def array_proxy__hash__(proxy):
    if proxy.issec:
        return hash(proxy.section)
    else:
        return hash((proxy.aid, proxy.elemIdx))

def array_getsecproxy(proxy, sinfo):
    if proxy.issec:
        secproxy = proxy.__class__(proxy.aid, proxy.ndims)
    else:
        secproxy = proxy.__secproxyclass__(proxy.aid, proxy.ndims)
    secproxy.section = sinfo
    return secproxy

def arraysecproxy__getstate__(proxy):
    return (proxy.aid, proxy.ndims, proxy.elemIdx, proxy.section)

def arraysecproxy__setstate__(proxy, state):
    proxy.aid, proxy.ndims, proxy.elemIdx, proxy.section = state

def array_proxy_elem(proxy, idx):  # array proxy [] overload method
    ndims = proxy.ndims
    isslice = True
    idxtype = type(idx)
    if idxtype == int:
        idx = (idx,)
        isslice = False
    elif idxtype == slice:
        idx = (idx,)
    assert len(idx) == ndims, "Dimensions of index " + str(idx) + " don't match array dimensions"
    if not isslice or not isinstance(idx[0], slice):
        proxy_clone = proxy.__class__(proxy.aid, ndims)
        proxy_clone.elemIdx = tuple(idx)
        return proxy_clone
    else:
        for _slice in idx:
            assert _slice.start is not None and _slice.stop is not None, 'Must specify start and stop indexes for array slicing'
        return charm.split(proxy, 1, slicing=idx)[0]

def array_proxy_method_gen(ep, argcount, argnames, defaults):  # decorator, generates proxy entry methods
    def proxy_entry_method(proxy, *args, **kwargs):
        num_args = len(args)
        if num_args < argcount and len(kwargs) > 0:
            args = list(args)
            for i in range(num_args, argcount):
                argname = argnames[i]
                # first look for argument in kwargs
                if argname in kwargs:
                    args.append(kwargs[argname])
                else:
                    # if not there, see if there is a default value
                    def_idx = i - argcount + len(defaults)
                    assert def_idx >= 0, 'Value not found for parameter \'' + argname + '\' of entry method'
                    args.append(defaults[def_idx])

        header = {}
        is_ray = kwargs.pop('is_ray', False)
        header['is_ray'] = is_ray
        blockFuture = None
        elemIdx = proxy.elemIdx
        if 'ret' in kwargs and kwargs['ret']:
            header[b'block'] = blockFuture = charm.Future()
            if elemIdx == ():
                header[b'bcast'] = header[b'bcastret'] = True
        elif 'awaitable' in kwargs and kwargs['awaitable']:
            header[b'block'] = blockFuture = charm.Future()
            if elemIdx == ():
                header[b'bcast'] = True
        if not proxy.issec or elemIdx != ():
            destObj = None
            aid = proxy.aid
            if Options.local_msg_optim and (len(args) > 0):
                array = charm.arrays[aid]
                if elemIdx in array:
                    destObj = array[elemIdx]
            if is_ray:
                blockFuture = charm.createFuture(store=True)
                args = list(args)
                args.append(blockFuture)
                args = tuple(args)
            msg = charm.packMsg(destObj, args, header)
            charm.CkArraySend(aid, elemIdx, ep, msg)
        else:
            root, sid = proxy.section
            header[b'sid'] = sid
            if Options.local_msg_optim and root == charm._myPe:
                charm.sectionMgr.thisProxy[root].sendToSectionLocal(sid, ep, header, *args)
            else:
                charm.sectionMgr.thisProxy[root].sendToSection(sid, ep, header, *args)
        return blockFuture
    proxy_entry_method.ep = ep
    return proxy_entry_method

def array_ckNew_gen(C, epIdx):
    @classmethod    # make ckNew a class (not instance) method of proxy
    def array_ckNew(cls, dims=None, ndims=-1, args=[], map=None, useAtSync=False, is_ray=False):
        # if charm.myPe() == 0: print("calling array ckNew for class " + C.__name__ + " cIdx=" + str(C.idx[ARRAY]))
        if type(dims) == int: dims = (dims,)

        if dims is None and ndims == -1:
            raise Charm4PyError('Bounds and number of dimensions for array cannot be empty in ckNew')
        elif dims is not None and ndims != -1 and ndims != len(dims):
            raise Charm4PyError('Number of bounds should match number of dimensions')
        elif dims is None and ndims != -1:  # create an empty array
            dims = (0,) * ndims

        # this is a restriction in Charm++. Charm++ won't tell you unless
        # error checking is enabled, resulting in obscure errors otherwise
        assert charm._myPe == 0, 'Cannot create arrays from PE != 0. Use charm.thisProxy[0].createArray() instead'

        map_gid = -1
        if map is not None:
            map_gid = map.gid

        header, creation_future = {}, None
        if sum(dims) > 0 and not charm.threadMgr.isMainThread():
            creation_future = charm.Future()
            header[b'block'] = creation_future
            header[b'bcast'] = True
            header[b'creation'] = True
            header[b'is_ray'] = is_ray

        msg = charm.packMsg(None, args, header)
        aid = charm.lib.CkCreateArray(C.idx[ARRAY], dims, epIdx, msg, map_gid, useAtSync)
        proxy = cls(aid, len(dims))
        if creation_future is not None:
            proxy.creation_future = creation_future
        return proxy
    return array_ckNew

def array_ckInsert_gen(epIdx):
    def array_ckInsert(proxy, index, args=[], onPE=-1, useAtSync=False, single=False, is_ray=False):
        if type(index) == int: index = (index,)
        assert len(index) == proxy.ndims, 'Invalid index dimensions passed to ckInsert'
        header = {}
        if single:
            header[b'single'] = True
            if not charm.threadMgr.isMainThread():
                proxy.creation_future = charm.Future()
                header[b'block'] = proxy.creation_future
                header[b'bcast'] = True
                header[b'creation'] = True
                header[b'is_ray'] = is_ray
        msg = charm.packMsg(None, args, header)
        charm.lib.CkInsert(proxy.aid, index, epIdx, onPE, msg, useAtSync)
    return array_ckInsert

def array_proxy_contribute(proxy, contributeInfo):
    charm.CkContributeToArray(contributeInfo, proxy.aid, proxy.elemIdx)

def arraysecproxy_contribute(proxy, contributeInfo):
    charm.CkContributeToSection(contributeInfo, proxy.section[1], proxy.section[0])

def array_proxy_doneInserting(proxy):
    charm.lib.CkDoneInserting(proxy.aid)


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
        if single:
            proxy = charm.proxyClasses[ARRAY][obj.__class__](aid, len(obj.thisIndex))
            obj.thisProxy = proxy[index]
        else:
            obj.thisProxy = charm.proxyClasses[ARRAY][obj.__class__](aid, len(obj.thisIndex))
        obj._contributeInfo = charm.lib.initContributeInfo(aid, obj.thisIndex, CONTRIBUTOR_TYPE_ARRAY)
        obj.migratable = True

    @classmethod
    def __baseEntryMethods__(cls):
        # 2nd method is used for 2 purposes:
        # - to register the migration constructor on Charm++ side (note that this migration constructor does nothing)
        # - Chare.migrated() is called whenever a chare has completed migration.
        #   The EntryMethod object with this name is used to profile Chare.migrated() calls.
        return ['__init__', 'migrated', 'AtSync']

    @classmethod
    def __getProxyClass__(C, cls, sectionProxy=False):
        if not sectionProxy:
            proxyClassName = cls.__name__ + 'ArrayProxy'
        else:
            proxyClassName = cls.__name__ + 'ArraySecProxy'
        M = dict()  # proxy methods
        entryMethods = charm.classEntryMethods[ARRAY][cls]
        for m in entryMethods:
            if m.epIdx == -1:
                raise Charm4PyError('Unregistered entry method')
            if m.name in {'__init__', 'migrated'}:
                continue
            argcount, argnames, defaults = getEntryMethodInfo(m.C, m.name)
            if Options.profiling:
                f = profile_send_function(array_proxy_method_gen(m.epIdx, argcount, argnames, defaults))
            else:
                f = array_proxy_method_gen(m.epIdx, argcount, argnames, defaults)
            f.__qualname__ = proxyClassName + '.' + m.name
            f.__name__ = m.name
            M[m.name] = f
        M['__init__'] = array_proxy_ctor
        M['__getitem__'] = array_proxy_elem
        M['__eq__'] = array_proxy__eq__
        M['__hash__'] = array_proxy__hash__
        M['ckNew'] = array_ckNew_gen(cls, entryMethods[0].epIdx)
        M['__getsecproxy__'] = array_getsecproxy
        M['ckInsert'] = array_ckInsert_gen(entryMethods[0].epIdx)
        M['ckDoneInserting'] = array_proxy_doneInserting
        if not sectionProxy:
            M['ckContribute'] = array_proxy_contribute  # function called when target proxy is Array
            M['__getstate__'] = array_proxy__getstate__
            M['__setstate__'] = array_proxy__setstate__
        else:
            M['ckContribute'] = arraysecproxy_contribute  # function called when target proxy is Array
            M['__getstate__'] = arraysecproxy__getstate__
            M['__setstate__'] = arraysecproxy__setstate__
        proxyCls = type(proxyClassName, (), M)  # create and return proxy class
        proxyCls.issec = sectionProxy
        return proxyCls

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
    global charm, Options, Reducer, Charm4PyError, CharmRemote, profile_send_function
    from .charm import charm, Charm4PyError, CharmRemote, profile_send_function
    Options = charm.options
    Reducer = charm.reducers
