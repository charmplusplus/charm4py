"""Standalone smoke test of charm4py._ckdyn, without the charm4py package.

Run:  CHARM_NOLOAD=1 python tests/thin/smoke_ckdyn.py +pe 1   (or +pe 2)

A minimal fake `charm` object provides the callables of SPEC-4a.md 2.1. The
main chare creates a group and exercises: argument conversion (int, float,
str, bytes, numpy, list with a nested numpy array, array.array, big ints,
None, bool, proxies, futures) on a broadcast (slow path) and a point-to-point
send (fast path); ret=True through a future (future_set and the future hook);
a built-in sum reduction to a group method; the foreign reducer to the main
chare and to a future; spawn/resume of a worker greenlet that pauses; and a
task that raises inside spawn. Each check prints a PASS line; the main chare
calls exit(0) once all reports are in.
"""
import array
import os
import sys
import traceback

import numpy as np
from greenlet import getcurrent

import charm4py._ckdyn as ck


def log(*a):
    print('[PE %d]' % ck.my_pe(), *a, flush=True)


def check(cond, what):
    if not cond:
        log('FAIL', what)
        sys.stdout.flush()
        ck.abort('smoke_ckdyn: FAIL ' + what)


class FakeFuture(object):
    def __init__(self, fid, src):
        self.fid = fid
        self.src = src
        self.store = False


def remote_future(pe, fid):
    return FakeFuture(fid, pe)


class Options(object):
    profiling = False
    remote_exec = False
    copy_received_arrays = os.environ.get('SMOKE_COPY_ARRAYS') == '1'


class FakeCharm(object):

    def __init__(self):
        self.classes = []          # (cls, kind, [method names], set of threaded names)
        self.ep_name = {}          # ep -> (cls, name)
        self.threaded = set()      # eps
        self.callbacks = {}        # (pe, fid) -> callable(value, rid)
        self.lastfid = 0

    # --- registration -------------------------------------------------
    def add(self, cls, kind, methods, threaded=()):
        self.classes.append((cls, kind, methods, set(threaded)))

    def registerMainModule(self):
        for cls, kind, methods, threaded in self.classes:
            entries = [(None, '__init__', cls.__name__ + '.__init__', 0, False, 0, (), ())]
            for name in methods:
                f = getattr(cls, name)
                argcount = f.__code__.co_argcount - 1
                argnames = tuple(f.__code__.co_varnames[1:argcount + 1])
                flags = ck.EP_THREADED if name in threaded else 0
                entries.append((f, name, cls.__name__ + '.' + name, flags, False, argcount, argnames,
                                f.__defaults__ or ()))
            idx, first_ep = ck.register_class(cls, kind, 0, entries)
            cls.idx = idx
            cls.eps = {}
            d = {}
            for i, name in enumerate(methods):
                ep = first_ep + 1 + i
                cls.eps[name] = ep
                self.ep_name[ep] = (cls, name)
                if name in threaded:
                    self.threaded.add(ep)
                f = getattr(cls, name)
                argcount = f.__code__.co_argcount - 1
                d[name] = ck.make_proxy_method(ep, argcount, f.__code__.co_varnames[1:argcount + 1],
                                               f.__defaults__ or (), name, cls.__name__ + 'Proxy.' + name)

            def elem(self, pe):
                p = type(self)(ck.GROUP, self.gid, 0)
                p._set_elem(pe)
                return p
            d['elem'] = elem
            pcls = type(cls.__name__ + 'Proxy', (ck.Proxy,), d)
            cls.Proxy = pcls
            ck.set_proxy_class(idx, pcls)

    # --- hooks ----------------------------------------------------------
    def _mainchare(self, target, argv):
        obj = Main.__new__(Main)
        obj._ckdyn_target = target
        obj._cond_next = None
        obj.thisProxy = Main.Proxy._from_target(target)
        obj.__init__(argv)
        return obj

    def _construct(self, target, ep, header, args):
        for cls, kind, methods, threaded in self.classes:
            if cls.idx == target.chare_idx:
                break
        obj = cls.__new__(cls)
        obj._ckdyn_target = target
        obj._cond_next = None
        obj.thisProxy = cls.Proxy(ck.GROUP, target.gid, 0)
        obj.thisIndex = target.pe
        obj.__init__(*args)
        return obj

    def _invoke(self, obj, ep, header, args):
        cls, name = self.ep_name[ep]
        f = getattr(cls, name)
        if ep in self.threaded:
            r = ck.spawn(ep, f, obj, *args)
            obj.spawn_results.append(r)
            return
        if name in HEADER_METHODS:
            n = f.__code__.co_argcount - 2      # the parameters before `header`
            res = f(obj, *args[:n], header=header)
        else:
            res = f(obj, *args)
        if b'block' in header:
            fut = header[b'block']
            ck.future_set(fut.src, fut.fid, res)

    def _process_exception(self, obj, ep, exc):
        traceback.print_exception(exc)
        sys.stdout.flush()
        ck.abort('smoke_ckdyn: exception in an entry method')

    def _deposit_future(self, fid, value, rid):
        cb = self.callbacks.pop((ck.my_pe(), fid))
        cb(value, rid)

    def _reduce_foreign(self, rid, values):
        check(rid == 7, 'foreign reducer id %r' % rid)
        out = []
        for v in values:
            out.extend(v)
        return sorted(out)

    def _pack_readonlies(self):
        return b'ro'

    def recvReadOnly(self, data):
        check(data == b'ro', 'readonly bytes %r' % data)

    def _destruct(self, obj):
        pass

    def _error(self, msg):
        sys.stderr.write('CkDyn error: ' + msg + '\n')
        sys.stderr.flush()

    def _pack(self, obj, reason):
        return None

    def Future(self):
        self.lastfid += 1
        return FakeFuture(self.lastfid, ck.my_pe())

    def createFuture(self, store=False):
        return self.Future()


charm = FakeCharm()
HEADER_METHODS = {'recv', 'fdone'}


class Main(object):

    def __init__(self, argv):
        n = ck.num_pes()
        self.n = n
        self.reports = {}
        # expected reports: tag -> count
        self.expected = {'recv-bcast': n, 'recv-p2p': 1, 'ret': 1, 'sum': n, 'foreign-main': 1,
                         'foreign-future': 1, 'coro': n, 'spawn-exc': n, 'spawn-done': n}
        log('PASS main chare started, %d PEs, refnum %d bytes' % (n, ck.refnum_bytes()))
        gid = ck.create_group(G.idx, (self.thisProxy,), 0, None, None)
        g = G.Proxy(ck.GROUP, gid, 0)
        check(g.elemIdx == -1 and g.gid == gid and not g.issec, 'group proxy attributes')
        self.g = g

        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        farr = np.asfortranarray(np.arange(6, dtype=np.int32).reshape(2, 3))
        lst = [1, 'two', np.arange(5, dtype=np.int16), {'k': (3, 4)}]
        aa = array.array('i', [5, -6, 7])
        ff = charm.Future()
        args = (42, 2.5, 'héllo', b'\x00bytes', arr, farr, lst, aa, 2 ** 70, 2 ** 63 + 5, None, True,
                self.thisProxy, g, ff, -(2 ** 40))
        g.recv(*args, writable=False)                           # broadcast: slow path
        g.elem(n - 1).recv(*args, writable=True)                # point-to-point: fast path

        f = g.elem(0).square(7, ret=True)
        check(type(f) is FakeFuture and f.src == 0, 'ret=True returns a future')
        charm.callbacks[(0, f.fid)] = self.got_square

        g.do_sum()
        fut = charm.Future()
        charm.callbacks[(0, fut.fid)] = self.got_foreign_future
        g.do_foreign(fut)
        g.coro_start()

    def got_square(self, value, rid):
        check(value == 49 and rid == -1, 'ret value %r rid %r' % (value, rid))
        log('PASS ret=True through a future: square(7) =', value)
        self.report('ret')

    def got_foreign_future(self, value, rid):
        check(value == list(range(10, 10 + self.n)) * 1 and rid == 7, 'foreign to future %r rid %r' % (value, rid))
        log('PASS foreign reducer to a future:', value, 'rid', rid)
        self.report('foreign-future')

    def fdone(self, value, header=None):
        check(header.get(b'custom_reducer') == 7, 'custom_reducer header %r' % header)
        check(value == list(range(self.n)), 'foreign result %r' % (value,))
        log('PASS foreign reducer to the main chare:', value)
        self.report('foreign-main')

    def report(self, tag):
        self.reports[tag] = self.reports.get(tag, 0) + 1
        check(self.reports[tag] <= self.expected[tag], 'too many reports of ' + tag)
        if self.reports == self.expected:
            log('PASS all %d reports received; exit(0)' % sum(self.expected.values()))
            ck.exit(0)


class G(object):

    def __init__(self, main):
        check(isinstance(main, Main.Proxy), 'constructor argument is a Main proxy')
        self.main = main
        self.spawn_results = []
        self.paused = None

    def recv(self, i, f, s, b, arr, farr, lst, aa, big, u64, none, flag, mp, gp, fut, neg, writable, header=None):
        bcast = header is not None and header.get(b'bcast', False)
        tag = 'recv-bcast' if bcast else 'recv-p2p'
        check(bcast == (not writable), tag + ': bcast header %r writable %r args %r' % (header, writable, (i, f, s, b, aa, big, u64, neg, mp, gp, fut.fid)))
        check(type(i) is int and i == 42, tag + ' int %r %r %r' % (i, f, s))
        check(type(f) is float and f == 2.5, tag + ' float')
        check(type(s) is str and s == 'héllo', tag + ' str')
        check(type(b) is bytes and b == b'\x00bytes', tag + ' bytes')
        check(type(arr) is np.ndarray and arr.dtype == np.float64 and arr.shape == (3, 4)
              and (arr == np.arange(12).reshape(3, 4)).all(), tag + ' ndarray')
        if not Options.copy_received_arrays:
            check(arr.flags.writeable == writable, tag + ' ndarray writable=%r' % arr.flags.writeable)
            check(isinstance(arr.base, ck.MsgRef), tag + ' ndarray base is a MsgRef')
        check(farr.flags.f_contiguous and farr.dtype == np.int32
              and (farr == np.arange(6).reshape(2, 3)).all(), tag + ' fortran ndarray')
        check(type(lst) is list and lst[0] == 1 and lst[1] == 'two' and lst[3] == {'k': (3, 4)}
              and type(lst[2]) is np.ndarray and (lst[2] == np.arange(5)).all() and lst[2].dtype == np.int16,
              tag + ' list with a nested ndarray: %r' % (lst,))
        check(type(aa) is array.array and aa.typecode == 'i' and list(aa) == [5, -6, 7], tag + ' array.array')
        check(big == 2 ** 70 and u64 == 2 ** 63 + 5 and neg == -(2 ** 40), tag + ' big ints')
        check(none is None and flag is True, tag + ' None/bool')
        check(type(mp) is Main.Proxy and mp == self.main, tag + ' main proxy round trip')
        check(type(gp) is G.Proxy and gp == self.thisProxy and gp.elemIdx == -1, tag + ' group proxy round trip')
        check(type(fut) is FakeFuture and fut.src == 0, tag + ' future slot')
        log('PASS', tag, 'arguments')
        self.main.report(tag)

    def square(self, x):
        return x * x

    def do_sum(self):
        types = {name: t for name, t, elem in ck.reducers()}
        ck.contribute(self, np.array([self.thisIndex + 1], dtype=np.int32), types['sum_int'],
                      self.thisProxy.sum_done)

    def sum_done(self, total):
        n = ck.num_pes()
        check(type(total) is int and total == n * (n + 1) // 2, 'sum result %r' % (total,))
        log('PASS built-in sum reduction to a group method:', total)
        self.main.report('sum')

    def do_foreign(self, fut):
        ft = ck.foreign_reducer_type()
        ck.contribute(self, (7, [self.thisIndex]), ft, self.main.fdone)
        ck.contribute(self, (7, [10 + self.thisIndex]), ft, fut)

    def coro_start(self):
        # runs on the main greenlet (slow path through _invoke, THREADED)
        pass

    def coro_wake(self, v):
        check(self.paused is not None, 'a paused greenlet to resume')
        ep_of_wake = ck.last_ep()   # spawn(ep, fn, obj, ...): the entry the task is charged to
        done = ck.resume(self.paused, v)
        check(done is True, 'resume returns True when the task finishes')
        check(self.spawn_results == [False], 'spawn returned False for the paused task: %r' % self.spawn_results)
        self.main.report('coro')
        # a task that raises: spawn re-raises it
        try:
            ck.spawn(ep_of_wake, _raise_task, self, 3)
            check(False, 'spawn of a raising task did not raise')
        except ValueError as e:
            check(str(e) == 'boom 3', 'exception text')
            log('PASS spawn re-raises the task exception')
            self.main.report('spawn-exc')
        # a task that finishes without pausing, on a pooled worker
        r = ck.spawn(ep_of_wake, _plain_task, self)
        check(r is True and self.plain_done, 'spawn of a task that does not pause returns True')
        log('PASS spawn of a non-pausing task on a pooled worker')
        self.main.report('spawn-done')


def _raise_task(obj, x):
    raise ValueError('boom %d' % x)


def _plain_task(obj):
    obj.plain_done = True


def coro_start(self):
    # executed inside a worker greenlet (THREADED entry)
    gr = getcurrent()
    main_gr = ck.main_greenlet()
    check(gr is not main_gr and main_gr is not None, 'running on a worker greenlet')
    self.paused = gr
    self.thisProxy.elem(self.thisIndex).coro_wake(5)
    value = main_gr.switch()          # pause
    check(value == 5, 'resumed with %r' % (value,))
    log('PASS spawn/resume: worker paused and was resumed with', value)


G.coro_start = coro_start

charm.add(Main, ck.MAINCHARE, ['fdone', 'report'])
charm.add(G, ck.GROUP, ['recv', 'square', 'do_sum', 'sum_done', 'do_foreign', 'coro_start', 'coro_wake'],
          threaded=['coro_start'])

if __name__ == '__main__':
    ck.init(charm, Options(), future_type=FakeFuture, remote_future=remote_future)
    ck.start(sys.argv)
