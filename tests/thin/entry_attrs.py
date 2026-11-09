"""Entry-method attributes, @entry(...) (SPEC-7-api.md section 2).

Run: python -m charmrun.start tests/thin/entry_attrs.py +p1   (and +p2, +p4)

Phases:
- register: every attribute lands in the CkDyn entry flags (_ckdyn.ep_flags) of the
  group and array registrations of a class; @entry composes with @coro and @when in
  either order, and is found through a __wrapped__ chain.
- errors: charm.register raises Charm4PyError for an unknown attribute, for inline
  on a coroutine, and for immediate or exclusive on a class registered as Group,
  Array or singleton Chare.
- deliver: expedited, notrace and appwork entries of a group and an array deliver,
  locally and to the last PE; an inline entry of a group branch and of an array
  element, sent to the same object from a coroutine, has run when the send returns
  (its side effect is visible right after the call, and not after a plain send to the
  same object); an inline send to another PE delivers.
The notrace attribute also removes the entry from Projections logs; that is checked
with the trace build (logs/step-7/trace/), not here.
"""
import functools
from charm4py import charm, Chare, Group, Array, coro, when, entry
from charm4py import _ckdyn
from charm4py.charm import Charm4PyError
from charm4py.chare import GROUP, ARRAY

EP_THREADED = _ckdyn.EP_THREADED
F = {'inline': _ckdyn.EP_INLINE, 'expedited': _ckdyn.EP_EXPEDITED,
     'immediate': _ckdyn.EP_IMMEDIATE, 'notrace': _ckdyn.EP_NOTRACE,
     'appwork': _ckdyn.EP_APPWORK, 'exclusive': _ckdyn.EP_EXCLUSIVE}
ATTR_MASK = sum(F.values())


def check(cond, what):
    if not cond:
        charm.abort('FAIL ' + what)
    print('[PE %d] PASS %s' % (charm.myPe(), what))


def plain_wrapper(func):
    """ a decorator that keeps only __wrapped__ (no copy of the function's attributes) """
    def w(*args, **kwargs):
        return func(*args, **kwargs)
    w.__wrapped__ = func
    w.__name__ = func.__name__
    return w


class Attrs(Chare):

    def __init__(self):
        self.inl_val = None
        self.ready = False

    @entry(expedited=True)
    def exp(self, x):
        return ('exp', charm.myPe(), x)

    @entry(inline=True)
    def inl(self, x):
        self.inl_val = x
        return ('inl', charm.myPe(), x)

    @entry(notrace=True)
    def nt(self, x):
        return ('nt', charm.myPe(), x)

    @entry(appwork=True)
    def aw(self, x):
        return ('aw', charm.myPe(), x)

    @entry(expedited=True, notrace=True, appwork=True)
    def multi(self, x):
        return x

    @entry(expedited=False)
    def off(self, x):
        return x

    @entry(expedited=True)
    @coro
    def entry_coro(self, x):
        return x

    @coro
    @entry(expedited=True)
    def coro_entry(self, x):
        return x

    @when('self.ready')
    @entry(notrace=True)
    def when_entry(self, x):
        return x

    @plain_wrapper
    @entry(appwork=True)
    def wrapped(self, x):
        return x

    def setready(self):
        self.ready = True

    def setval(self, x):
        self.inl_val = x

    @coro
    def inline_self(self, x):
        """ an inline send to this object runs before the send returns; a plain one
            does not (the control) """
        self.inl_val = None
        self.thisProxy[self.thisIndex].setval(x)
        before = self.inl_val
        self.thisProxy[self.thisIndex].inl(x)
        return before, self.inl_val


def ep_flags(C, kind, name):
    for em in charm.classEntryMethods[kind][C]:
        if em.name == name:
            return _ckdyn.ep_flags(em.epIdx)
    raise KeyError(name)


def make_bad(attrs, coro_too=False):
    class Bad(Chare):
        def m(self):
            pass
    f = Bad.m
    if coro_too:
        f = coro(f)
    Bad.m = entry(**attrs)(f)
    return Bad


class Main(Chare):

    def __init__(self, args):
        self.run()

    @coro
    def run(self):
        self.phase_register()
        self.phase_errors()
        self.phase_deliver()
        print('entry_attrs: all phases passed')
        charm.exit()

    def phase_register(self):
        for kind, kname in ((GROUP, 'group'), (ARRAY, 'array')):
            for name, attr in (('exp', 'expedited'), ('inl', 'inline'), ('nt', 'notrace'), ('aw', 'appwork')):
                fl = ep_flags(Attrs, kind, name)
                check(fl & ATTR_MASK == F[attr],
                      'register %s: %s has %s only (flags 0x%x)' % (kname, name, attr, fl))
            fl = ep_flags(Attrs, kind, 'multi')
            want = F['expedited'] | F['notrace'] | F['appwork']
            check(fl & want == want, 'register %s: three attributes on one entry (0x%x)' % (kname, fl))
            check(ep_flags(Attrs, kind, 'off') & F['expedited'] == 0,
                  'register %s: expedited=False sets nothing' % kname)
            for name in ('entry_coro', 'coro_entry'):
                fl = ep_flags(Attrs, kind, name)
                check(fl & EP_THREADED and fl & F['expedited'],
                      'register %s: %s is threaded and expedited (0x%x)' % (kname, name, fl))
            check(ep_flags(Attrs, kind, 'when_entry') & F['notrace'], 'register %s: @when over @entry' % kname)
            check(ep_flags(Attrs, kind, 'wrapped') & F['appwork'], 'register %s: @entry under a __wrapped__ wrapper' % kname)
            check(ep_flags(Attrs, kind, 'setready') & ATTR_MASK == 0, 'register %s: a plain entry has no attribute flags' % kname)

    def phase_errors(self):
        cases = [({'bogus': True}, False, (GROUP,), 'unknown attribute'),
                 ({'inline': True}, True, (GROUP,), 'inline on a coroutine'),
                 ({'immediate': True}, False, (GROUP,), 'immediate on a group'),
                 ({'exclusive': True}, False, (ARRAY,), 'exclusive on an array')]
        for attrs, coro_too, kinds, what in cases:
            Bad = make_bad(attrs, coro_too)
            try:
                charm.register(Bad, kinds)
            except Charm4PyError as e:
                check(Bad not in charm.registered, 'errors: %s raises Charm4PyError (%s)' % (what, e))
            else:
                charm.abort('FAIL errors: %s was accepted' % what)

    def phase_deliver(self):
        last = charm.numPes() - 1
        g = Group(Attrs)
        a = Array(Attrs, charm.numPes())
        for proxy, kname in ((g, 'group'), (a, 'array')):
            for pe in sorted({0, last}):
                for name in ('exp', 'nt', 'aw', 'inl'):
                    r = getattr(proxy[pe], name)(7, ret=True).get()
                    check(r == (name, pe, 7), '%s %s send to %d delivers: %r' % (kname, name, pe, r))
            r = proxy.multi(5, ret=True).get()
            check(r == [5] * charm.numPes(), '%s broadcast to a multi-attribute entry: %r' % (kname, r))
            proxy.when_entry(3)
            proxy.setready(awaitable=True).get()
            for pe in sorted({0, last}):
                r = proxy[pe].inline_self(pe + 11, ret=True).get()
                check(r == (None, pe + 11), '%s inline send to self on PE %d has run when the send returns (%r)'
                      % (kname, pe, r))


charm.start(Main)
