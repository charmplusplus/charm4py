from . import charm, Chare, Group, Reducer, when
from collections import defaultdict


# Reduction Info object: holds state for an in-progress reduction
class RedInfo(object):
    def __init__(self):
        self.ready = False  # got all messages, can reduce and send contribution to the parent
        self.msgs = []  # list of reduction msgs received on this PE
        self.reducer = None  # reducer function
        self.cb = None  # reduction callback


class SectionManager(Chare):

    class SectionEntry(object):

        def __init__(self):
            self.final = False  # section creation completed at this PE
            self.parent = None
            self.children = []  # these are PE numbers
            self.local_elems = []  # list of local chares that are part of the section
            self.buffered_msgs = []  # stores msgs received for this section before creation has completed
            self.redno = 0  # current reduction number for this section
            self.reds = []  # list of RedInfo objects for pending reductions


    def __init__(self):
        assert not hasattr(charm, 'sectionMgr')
        charm.sectionMgr = self
        self.profiling = charm.options.profiling
        self.sections = defaultdict(SectionManager.SectionEntry)  # stores section entries for this PE
        self.send_ep = self.thisProxy.sendToSection.ep

    def createSectionDown(self, sid, pes, parent=None):
        entry = self.sections[sid]
        entry.final = True
        assert len(entry.local_elems) > 0
        mype = charm.myPe()
        if parent is None:
            # I'm the root
            pes.discard(mype)
            pes = [mype] + sorted(pes)
        else:
            entry.parent = self.thisProxy[parent]
        subtrees = charm.getTopoSubtrees(mype, pes, bfactor=4)
        for subtree in subtrees:
            child = self.thisProxy[subtree[0]]
            child.createSectionDown(sid, subtree, mype)
            entry.children.append(child.elemIdx)

        for ep, header, args in entry.buffered_msgs:
            self.sendToSectionLocal(sid, ep, header, *args)
        entry.buffered_msgs = []
        self.releaseRed(sid, entry, entry.reds)

    @when('cons is not None or gid in charm.groups')
    def createGroupSectionDown(self, sid, gid, pes, parent=None, cons=None):
        entry = self.sections[sid]
        entry.final = True
        mype = charm.myPe()
        if parent is None:
            # I'm the root
            pes.discard(mype)
            pes = [mype] + sorted(pes)
        else:
            entry.parent = self.thisProxy[parent]

        if cons is not None and parent is None:
            cons[0] = self.thisIndex  # cons[0] used to store the section root

        subtrees = charm.getTopoSubtrees(mype, pes, bfactor=4)
        for subtree in subtrees:
            child = self.thisProxy[subtree[0]]
            child.createGroupSectionDown(sid, gid, subtree, mype, cons)
            entry.children.append(child.elemIdx)

        if cons is not None:
            # creating constrained groups
            root, ep, header, args = cons
            em = charm.entryMethods[ep]
            obj = object.__new__(em.C)  # create object but don't call __init__
            Group.initMember(obj, gid)
            super(em.C, obj).__init__()  # call Chare class __init__ first
            obj.thisProxy = obj.thisProxy.__getsecproxy__((root, sid))
            entry.local_elems = [obj]
            em.run(obj, header, args)  # now call the user's __init__
            charm.groups[gid] = obj
            if gid in charm.groupMsgBuf:
                for ep, header, args in charm.groupMsgBuf[gid]:
                    charm.invokeEntryMethod(obj, ep, header, args)
                del charm.groupMsgBuf[gid]
        else:
            entry.local_elems = [charm.groups[gid]]

        for ep, header, args in entry.buffered_msgs:
            self.sendToSectionLocal(sid, ep, header, *args)
        entry.buffered_msgs = []
        self.releaseRed(sid, entry, entry.reds)

    # called locally
    def sendToSectionLocal(self, sid, ep, header, *args):
        entry = self.sections[sid]
        if not entry.final:
            entry.buffered_msgs.append((ep, header, args))
            return

        if len(entry.children) > 0:
            profiling = self.profiling
            if profiling:
                em = charm.runningEntryMethod
                em.startMeasuringSendTime()
            msg = charm.packMsg(None, [sid, ep, header] + list(args), {})
            charm.lib.CkGroupSendMulti(self.thisProxy.gid, entry.children,
                                       self.send_ep, msg)
            del msg
            if profiling:
                em.stopMeasuringSendTime()

        for obj in entry.local_elems:
            charm.invokeEntryMethod(obj, ep, header, args)

    def sendToSection(self, sid, ep, header, *args):
        entry = self.sections[sid]
        if not entry.final:
            entry.buffered_msgs.append((ep, header, args))
            return

        if len(entry.children) > 0:
            profiling = self.profiling
            if profiling:
                em = charm.runningEntryMethod
                em.startMeasuringSendTime()
            # SectionManagerExt in Charm++ has a pointer to the multicast message,
            # this tells it to forward the msg to the children using CkSendMsgBranchMulti
            # (thus avoiding any copies)
            charm.lib.sendToSection(self.thisProxy.gid, entry.children)
            if profiling:
                charm.recordSend(charm.msg_recv_stats[4])  # send size is same as last received msg size
                em.stopMeasuringSendTime()

        for obj in entry.local_elems:
            charm.invokeEntryMethod(obj, ep, header, args)

    def contrib(self, sid, redno, data, reducer, cb):
        entry = self.sections[sid]
        idx = redno - entry.redno
        reds = entry.reds
        for _ in range(idx + 1 - len(reds)):
            reds.append(RedInfo())
        redinfo = reds[idx]
        redinfo.msgs.append(data)
        if cb is not None:
            redinfo.cb = cb
        if reducer is not None:
            redinfo.reducer = reducer
        if not entry.final:
            return
        if len(redinfo.msgs) == len(entry.children) + len(entry.local_elems):
            redinfo.ready = True
            if idx == 0:
                self.releaseRed(sid, entry, reds)

    def releaseRed(self, sid, entry, reds):
        while len(reds) > 0:
            redinfo = reds[0]
            if redinfo.ready:
                reds.pop(0)
                entry.redno += 1
                reducer = redinfo.reducer
                if reducer is None:  # empty reduction
                    if entry.parent is None:
                        redinfo.cb(None)
                    else:
                        entry.parent.contrib(sid, entry.redno - 1, None, None, None)
                else:
                    reduced_data = reducer(redinfo.msgs)
                    if entry.parent is None:
                        # reached the root, send result to callback
                        if reducer.hasPostprocess:
                            reduced_data = reducer.postprocess(reduced_data)
                        redinfo.cb(reduced_data)
                    else:
                        if reducer == Reducer._bcast_exc_reducer:
                            entry.parent.contrib(sid, entry.redno - 1, reduced_data, reducer, None)
                        else:
                            entry.parent.contrib(sid, entry.redno - 1, reduced_data, None, None)
            else:
                return


def _sectionloc(contributions):
    numsections = len(contributions[0])
    result = [set() for _ in range(numsections)]
    for c in contributions:
        for i, pes in enumerate(c):
            result[i].update(pes)
    return result


Reducer.addReducer(_sectionloc)
