"""Wire formats of the load replies that charm4py.appspector sends.

Pure Python (struct only, no charm import), so it can be tested without a
running job. The layouts are those of Charm++'s own handlers, so a client
that reads those (appSpector's ccsio.objload and ccsio.perf) reads these
unchanged. Every number is little-endian.

Per-object loads (handler "charm4py ObjLoads", the layout of Charm++'s
"CkPerfObjLoads" and "LBObjLoads"):

    int32 numPes
    per PE:
        int32 pe, double wallNow, double elapsed, double idleWall, int32 n
        n x { uint64 objId, int32 om, double objWall, uint8 migratable }

    elapsed is the time since the PE started measuring and idleWall the part
    of it not spent in entry methods, both cumulative; objWall is the
    cumulative time of one chare. A client takes differences of two replies.
    om/objId name the chare: arrays (om = array id, objId = index packed by
    obj_id()), groups (om = group id, objId = PE), singleton chares and the
    mainchare (om = -1, objId from the chare id).

Utilization by entry method (handler "charm4py SumDetail", the layout of
"CkPerfSumDetail compressed"):

    int32 numBins                  0: nothing to report yet, reply ends here
    int32 numProcs
    per bin:
        int16 count
        count x { int16 ep, uint8 util }     util = percent * 2.5, max 250

    charm4py sends one bin per reply: the mean over PEs of each entry
    method's share of the wall time since the previous reply.

Entry-method names (handler "charm4py SumDetail sts"): text in the format of
a Charm++ .sts file, one line per entry method:

    ENTRY CHARE <ep> "<Class>.<method>" 0 0
"""

import struct
import zlib

OTHER_EP = 10000        # Charm++'s "below the cutoff" bucket
TIMER_EP = 10001        # callables run by charm.scheduleCallableAfter
RESUME_EP = 10002       # a coroutine resumed whose entry method is unknown
PSEUDO_EPS = {OTHER_EP: "other", TIMER_EP: "charm.scheduled callables",
              RESUME_EP: "resumed coroutine"}
UTIL_SCALE = 2.5
MAX_EPS_PER_BIN = 64    # the rest are folded into OTHER_EP

_MASK64 = (1 << 64) - 1
_HDR = struct.Struct("<idddi")
_OBJ = struct.Struct("<QidB")


def obj_id(index):
    """A deterministic uint64 for an array index tuple: the coordinates
    packed (1 to 3 dimensions of non-negative ints that fit), else a CRC of
    their repr. Never hash(): it differs between processes, and a client
    follows an element across PEs by this id."""
    n = len(index)
    try:
        if n == 1 and 0 <= index[0] <= _MASK64:
            return index[0]
        if n == 2 and 0 <= index[0] < 1 << 32 and 0 <= index[1] < 1 << 32:
            return (index[0] << 32) | index[1]
        if n == 3 and all(0 <= i < 1 << 21 for i in index):
            return (index[0] << 42) | (index[1] << 21) | index[2]
    except TypeError:
        pass
    data = repr(tuple(index)).encode()
    return ((zlib.crc32(data) << 32) | zlib.adler32(data)) & _MASK64


def chare_obj_id(cid):
    """objId of a singleton chare, cid = (pe, object pointer)."""
    pe, ptr = cid
    return (((pe & 0xFFFF) << 48) ^ ptr) & _MASK64


def encode_objloads(pes):
    """pes: iterable of (pe, now, elapsed, idle, objs) with objs a list of
    (om, objId, seconds, migratable)."""
    pes = list(pes)
    out = [struct.pack("<i", len(pes))]
    for pe, now, elapsed, idle, objs in pes:
        out.append(_HDR.pack(pe, now, elapsed, idle, len(objs)))
        for om, oid, secs, mig in objs:
            out.append(_OBJ.pack(oid & _MASK64, om, secs, 1 if mig else 0))
    return b"".join(out)


def encode_sumdetail(num_procs, bins):
    """bins: list of lists of (ep, percent). An empty list encodes the
    "nothing yet" reply (numBins 0)."""
    if not bins:
        return struct.pack("<i", 0)
    out = [struct.pack("<ii", len(bins), num_procs)]
    for b in bins:
        cells = []
        for ep, pct in b:
            u = int(round(pct * UTIL_SCALE))
            if u > 0 and -32768 <= ep < 32768:
                cells.append(struct.pack("<hB", ep, min(250, u)))
        out.append(struct.pack("<h", len(cells)))
        out.extend(cells)
    return b"".join(out)


def ep_percentages(prev, cur, num_pes):
    """Mean percent of wall time per entry method between two gathers.

    prev, cur: {pe: (now, {ep: cumulative seconds})}. A PE's share is its
    growth over its own wall interval; the mean is over num_pes (a PE missing
    from either gather counts as idle). Beyond MAX_EPS_PER_BIN entries the
    smallest are folded into OTHER_EP."""
    acc = {}
    for pe, (now, eps) in cur.items():
        p = prev.get(pe)
        if p is None or now <= p[0]:
            continue
        dt = now - p[0]
        peps = p[1]
        for ep, secs in eps.items():
            d = secs - peps.get(ep, 0.0)
            if d > 0:
                acc[ep] = acc.get(ep, 0.0) + 100.0 * min(1.0, d / dt)
    n = max(1, num_pes)
    items = sorted(((ep, v / n) for ep, v in acc.items()), key=lambda kv: -kv[1])
    if len(items) > MAX_EPS_PER_BIN:
        rest = sum(v for _, v in items[MAX_EPS_PER_BIN - 1:])
        items = items[:MAX_EPS_PER_BIN - 1] + [(OTHER_EP, rest)]
    return items


def sts_text(names):
    """names: {ep: "Class.method"}; the pseudo entries are added."""
    all_names = {ep: n for ep, n in PSEUDO_EPS.items() if ep != OTHER_EP}
    all_names.update(names)
    lines = ['ENTRY CHARE %d "%s" 0 0' % (ep, all_names[ep].replace('"', "'"))
             for ep in sorted(all_names)]
    return "\n".join(lines) + "\n"
