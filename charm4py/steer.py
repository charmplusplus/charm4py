"""Steering a running charm4py program over CCS.

A program exposes named Python callables; a CCS client (appSpector's
``ccsio.steer``, or anything that speaks CCS) calls them by name while
the job runs::

    from charm4py import charm, Chare, Array, steer

    @steer.expose
    def set_rate(rate: float):
        "Set the diffusion rate on every chare."
        cells.set_rate(rate, awaitable=True).get()   # broadcast and wait
        return rate

    def main(args):
        global cells
        cells = Array(Cell, 16)
        steer.init()                                  # CCS handler "c4pSteer" on PE 0
        ...

    charm.start(main)

Run the program with ``++server ++server-port N`` (and, for anything but a
private test, ``++server-auth KEYFILE``). Import this module before
``charm.start`` (at the top of the program), so that its chare type is
registered on every process.

Wire format
-----------
The CCS handler ``c4pSteer`` lives on PE 0. A request is UTF-8 JSON
(trailing NUL bytes are ignored)::

    {"call": "set_rate", "args": [0.5], "kwargs": {}}

The reply is UTF-8 JSON, always exactly one per request::

    {"ok": true,  "result": <the callable's return value>}
    {"ok": false, "error": "ValueError: rate must be positive", "traceback": "..."}

Built-in calls (reserved names):

``list``
    result ``{"calls": [{"name", "signature", "doc"}, ...], "eval": bool}``.
``eval``
    ``args: [source]``: run Python source on PE 0 (an expression gives its
    value; statements run, and if the last one is an expression its value
    is the result). Off unless enabled, see Security below.

Return values are converted to JSON: NumPy arrays and scalars (real
``numpy.ndarray`` / ``numpy.generic`` instances only) become lists and
numbers, tuples become lists, sets sorted lists, bytes text, dict keys
strings (``str(key)``), and NaN / Infinity / -Infinity become ``null`` (the
reply is strict JSON). Anything else JSON cannot represent becomes its
``repr``.

Execution
---------
The CCS handler runs on PE 0 inside the Charm++ scheduler, on the main
thread, where a charm4py future cannot be waited on. So the handler only
parses the request, takes a delayed-reply token (``CcsDelayReply``) and
sends a message to a steering chare on PE 0; the call itself runs there in
a coroutine (a ``@coro`` entry method). A callable may therefore block:
``proxy.method(awaitable=True).get()``, ``ret=True`` broadcasts,
``charm.sleep``, ``charm.allReduce`` ... While it waits, PE 0 keeps
scheduling other work, including other CCS requests (liveViz images, other
steering calls). If the callable returns a charm4py Future, the reply waits
for that future's value. When the call returns, its result goes back with
``CcsSendDelayedReply``. A callable that blocks PE 0 without yielding
(``time.sleep``, a long computation) blocks PE 0 for that long, as any
entry method would. ``list``, malformed requests and refused requests
are answered at once by the handler.

At most ``max_inflight`` calls (``steer.init``, default 64) run at a time;
a request beyond that is answered at once with
``{"ok": false, "error": "busy"}``. With ``steer.init(timeout=SECONDS)`` a
call that has not finished after that long is answered with a timeout
error; the call itself cannot be stopped and keeps running (and keeps its
in-flight slot) until it returns, and its result is then dropped, so the
client still gets exactly one reply. Every accepted request gets exactly
one reply, also when the callable raises ``SystemExit``,
``KeyboardInterrupt`` or ``GeneratorExit`` (the reply is sent, then the
exception propagates) or when the result cannot be encoded.

Callables are looked up in the process that runs PE 0: expose them at
module level, or from code that runs on PE 0 (``main``). An ``expose``
call made only in another process (e.g. in a chare's constructor on
another PE) is not visible to the handler.

Security
--------
CCS with ``++server-auth KEYFILE`` authenticates every request: the client
proves it holds the key (SHA-1 challenge/response with a per-client
counter), and the server proves it back in each reply. It does NOT encrypt:
requests and replies, including steering arguments and results, cross the
network in the clear. Without ``++server-auth`` (or with a key file that
has no level-0 key, below), anyone who can reach the CCS port can call
every exposed function. Keep the port behind an ssh
forward or a private network, use ``++server-auth`` with a mode-600 key
file, and expose only what a key holder may do.

Charm++'s CCS server (``CcsSecMan_default`` in ``ccs-server.C``, the same
in Reconverse) reads the key file as lines ``LEVEL KEY`` and accepts an
unauthenticated request whenever the file holds no level-0 key, even with
``++server-auth``: an empty file, a file with only ``1 <hex>`` lines, or
only comments leaves the server open. ``steer`` therefore treats the
server as authenticating only if the key file it names has a level-0 key,
parsed the way Charm++ parses it (see ``keyfile_has_level0``).

``eval`` is remote code execution with the program's privileges for
anyone who has a key. It is off unless BOTH the program calls
``steer.init(allow_eval=True)`` AND PE 0's command line has
``++server-auth FILE`` (that exact form, a separate argument, as Charm++
takes it; ``++server-auth=FILE`` is not recognized by Charm++ and does not
count) with every such FILE readable by PE 0's process and holding a
level-0 key. Otherwise ``init`` prints why on PE 0 and eval stays off.
With a level-0 key, Charm++ rejects every unauthenticated request, so
every request that reaches the handler was authenticated, but at SOME
level that has a key in the file: the handler cannot see the level
(charm4py does not expose ``CcsSecAttr``), so a holder of any key in the
file, not only the level-0 key, can eval. Put only keys of people
allowed to eval in that file. The check reads the file when ``init``
runs, relative to the working directory at import of ``charm4py.steer``;
it cannot see a file changed since the server read it, nor a file that
only charmrun reads (when charmrun runs the CCS server, as with
``python -m charmrun.start ... ++local``, it may take ``++server-auth``
off the program's command line; eval then stays off even though the
server authenticates). Every eval request is printed on PE 0's stdout
(the first 200 characters of the source; all of it with ``verbose``).

Error replies carry a Python traceback only when the server authenticates
(the check above) or ``verbose=True``; otherwise just ``"Type: message"``.
Call arguments are printed only with ``verbose=True``.
"""

import ast
import inspect
import json
import math
import os
import re
import sys
import traceback

from .charm import charm, Chare, register
from .entry_method import coro


HANDLER = "c4pSteer"
RESERVED = ("list", "eval")
MAX_INFLIGHT = 64       # default bound on concurrently running calls

_registry = {}          # name -> callable (this process)
_agent = None           # proxy of the steering chare, set by init()
_future_types = ()      # charm4py Future classes, filled in by init()
_start_cwd = os.getcwd()    # relative key file names resolve against this


# -- exposing callables --------------------------------------------------------

def expose(fn=None, name=None):
    """Make ``fn`` callable by steering clients under ``name`` (default:
    its ``__name__``). Use as ``steer.expose(fn)``, ``steer.expose(fn,
    "name")``, ``@steer.expose``, ``@steer.expose("name")`` or
    ``@steer.expose(name="name")``. Returns ``fn`` unchanged."""
    if fn is None:
        return lambda f: expose(f, name)
    if isinstance(fn, str) and name is None:       # @steer.expose("name")
        return lambda f, _n=fn: expose(f, _n)
    if not callable(fn):
        raise TypeError("steer.expose needs a callable, got %r" % (fn,))
    name = name or fn.__name__
    if name in RESERVED:
        raise ValueError("'%s' is a built-in steering call; choose another name" % name)
    _registry[name] = fn
    return fn


def unexpose(name):
    """Remove an exposed callable (no error if it is not there)."""
    _registry.pop(name, None)


def exposed():
    """The exposed callables: a dict name -> callable (a copy)."""
    return dict(_registry)


# -- CCS authentication check -------------------------------------------------

# Charm++ (conv-ccs/ccs-server.C, CcsSecMan_default; Reconverse's copy is the
# same) reads the key file with fgets(line, 200, f), so a line longer than 199
# bytes is parsed as several pieces, and parses each piece with
# sscanf(line, "%d%s", &level, key). A piece with both items and
# 0 <= level < 255 sets keys[level] if the key starts with two hex digits
# (isxdigit) or with "OTP" (otherwise the server exits at startup).
# allowRequest_default lets an unauthenticated request through iff keys[0] is
# NULL. The regex is that sscanf: C-locale isspace, optional sign, digits,
# then the next run of non-space characters.
_FGETS_MAX = 199
_KEY_PIECE = re.compile(rb"[ \t\n\v\f\r]*([+-]?[0-9]+)[ \t\n\v\f\r]*([^ \t\n\v\f\r]+)")
_HEXDIGITS = b"0123456789abcdefABCDEF"


def _fgets_pieces(data):
    """The strings successive fgets(buf, 200, f) calls return, each cut at
    its first NUL (where sscanf stops reading)."""
    i, n = 0, len(data)
    while i < n:
        nl = data.find(b"\n", i, i + _FGETS_MAX)
        end = nl + 1 if nl >= 0 else min(i + _FGETS_MAX, n)
        yield data[i:end].split(b"\0", 1)[0]
        i = end


def keyfile_has_level0(data):
    """True if the contents of a CCS key file (bytes) give Charm++'s
    server a level-0 key, i.e. if the server refuses unauthenticated
    requests. A level that would overflow a C int is not taken as 0 (the
    safe side: eval stays off)."""
    for piece in _fgets_pieces(data):
        m = _KEY_PIECE.match(piece)
        if m is None or int(m.group(1)) != 0:
            continue
        key = m.group(2)
        if (len(key) >= 2 and key[0] in _HEXDIGITS and key[1] in _HEXDIGITS) \
                or key.startswith(b"OTP"):
            return True
    return False


def auth_status(argv=None):
    """(True, why) if PE 0's command line makes the CCS server refuse
    unauthenticated requests: ``++server-auth FILE`` (as separate arguments,
    the only form Charm++ accepts), and every such FILE is readable and has a
    level-0 key. Otherwise (False, why)."""
    argv = sys.argv if argv is None else argv
    files = []
    for i, a in enumerate(argv):
        if a == "++server-auth":
            if i + 1 >= len(argv):
                return False, "++server-auth is the last argument (no key file)"
            files.append(argv[i + 1])
    if not files:
        return False, "the job was started without ++server-auth FILE"
    for f in files:
        path = os.path.join(_start_cwd, f)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as e:
            return False, "cannot read the ++server-auth key file %r (%s)" % (f, e.strerror or e)
        if not keyfile_has_level0(data):
            return False, ("the ++server-auth key file %r has no level-0 key (a line "
                           "'0 <hex key>'), so the CCS server accepts unauthenticated "
                           "requests" % f)
    return True, "the CCS server requires a key (++server-auth with a level-0 key)"


def auth_enabled(argv=None):
    """True if the CCS server refuses unauthenticated requests (see
    ``auth_status``)."""
    return auth_status(argv)[0]


# -- setup -----------------------------------------------------------------------

def init(allow_eval=False, namespace=None, verbose=False, max_inflight=MAX_INFLIGHT,
         timeout=None):
    """Start serving steering requests: creates the steering chare on PE 0,
    which registers the CCS handler ``c4pSteer``. Call once, after
    ``charm.start``, in PE 0's process (in ``main``); later calls return
    the same proxy. Callables must be exposed in PE 0's process too (at
    module level or in ``main``).

    allow_eval: also accept ``eval`` requests, but only if the CCS server
        authenticates: ``++server-auth FILE`` whose FILE has a level-0 key
        (eval is remote code execution; see the module doc).
    namespace: the globals dict ``eval`` runs in (default: ``__main__``'s).
        Only used on PE 0's process; pass it from code running there.
    verbose: print one line on PE 0 per steering call, with its arguments
        (eval calls are always printed), and send tracebacks with errors
        even without authentication.
    max_inflight: most calls running at once; more are answered
        ``{"ok": false, "error": "busy"}``.
    timeout: seconds after which a call still running is answered with a
        timeout error (its late result is dropped); None: no timeout.
    """
    global _agent, _future_types
    if _agent is not None:
        return _agent
    if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight < 1:
        raise ValueError("max_inflight must be a positive int, got %r" % (max_inflight,))
    if timeout is not None:
        timeout = float(timeout)
        if not timeout > 0 or math.isinf(timeout):
            raise ValueError("timeout must be a positive number of seconds or None")
    from .threads import Future, CollectiveFuture
    _future_types = (Future, CollectiveFuture)
    if namespace is not None:
        SteerAgent.namespace = namespace
    _agent = Chare(SteerAgent, args=[bool(allow_eval), bool(verbose), max_inflight, timeout],
                   onPE=0)
    return _agent


# -- request handling (pure Python, no Charm++ calls) ---------------------------

def to_json(x, _active=None):
    """``x`` as plain JSON data (dict with str keys, list, str, int, finite
    float, bool, None). NumPy arrays and scalars (only real numpy.ndarray /
    numpy.generic instances, and only if numpy is already imported) become
    lists and numbers; tuples lists; sets sorted lists; bytes text; dict
    keys ``str(key)``; NaN and +-Infinity None (JSON null); anything else its
    ``repr``. ValueError on a circular structure."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, (np.ndarray, np.generic)):
        x = x.tolist() if isinstance(x, np.ndarray) else x.item()
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", "replace")
    if isinstance(x, (dict, list, tuple, set, frozenset)):
        _active = set() if _active is None else _active
        if id(x) in _active:
            raise ValueError("circular reference")
        _active.add(id(x))
        try:
            if isinstance(x, dict):
                return {(k if isinstance(k, str) else str(k)): to_json(v, _active)
                        for k, v in x.items()}
            if isinstance(x, (set, frozenset)):
                x = sorted(x, key=repr)
            return [to_json(v, _active) for v in x]
        finally:
            _active.discard(id(x))
    return repr(x)


_FALLBACK = b'{"ok": false, "error": "charm4py.steer: the reply could not be encoded"}'


def encode(reply):
    """A reply dict as the bytes sent to the client (strict JSON). Never
    raises: a reply that cannot be converted becomes an error reply."""
    try:
        return json.dumps(to_json(reply), allow_nan=False).encode("utf-8")
    except Exception as e:      # e.g. a circular structure, a failing repr
        try:
            return json.dumps({"ok": False, "error": "result is not JSON-serializable: %s: %s"
                               % (type(e).__name__, e)}).encode("utf-8")
        except Exception:
            return _FALLBACK


def error_reply(exc, with_traceback=False):
    try:
        text = "%s: %s" % (type(exc).__name__, exc)
    except Exception:
        text = type(exc).__name__
    r = {"ok": False, "error": text}
    if with_traceback and exc.__traceback__ is not None:
        try:
            r["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            pass
    return r


def _send(token, reply):
    """Send ``reply`` as the immediate reply (token None) or the delayed
    reply for ``token``. Never raises: if encoding or sending fails, it
    retries with a minimal error message, then gives up with a note on
    stdout."""
    msg = encode(reply)
    err = None
    for m in ((msg,) if msg is _FALLBACK else (msg, _FALLBACK)):
        try:
            if token is None:
                charm.CcsSendReply(m)
            else:
                charm.CcsSendDelayedReply(token, m)
            return
        except Exception as e:
            err = e
    print("charm4py.steer: could not send a reply: %s" % error_reply(err)["error"])


def parse_request(data):
    """CCS request bytes -> (name, args, kwargs); ValueError if malformed."""
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).rstrip(b"\x00").decode("utf-8")
    req = json.loads(data)
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    name = req.get("call")
    if not isinstance(name, str) or not name:
        raise ValueError('request needs "call": a name (try "list")')
    args = req.get("args", [])
    kwargs = req.get("kwargs", {})
    if not isinstance(args, list):
        raise ValueError('"args" must be a list')
    if not isinstance(kwargs, dict):
        raise ValueError('"kwargs" must be an object')
    return name, args, kwargs


def describe(registry=None):
    """The 'list' result's calls: name, signature and docstring of each."""
    registry = _registry if registry is None else registry
    out = []
    for name in sorted(registry):
        fn = registry[name]
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        out.append({"name": name, "signature": sig, "doc": inspect.getdoc(fn) or ""})
    return out


def run_source(source, namespace):
    """eval: an expression's value, or run statements and return the value of
    a trailing expression statement (None otherwise)."""
    tree = ast.parse(source, "<steer>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    if tree.body:
        exec(compile(tree, "<steer>", "exec"), namespace)
    if last is not None:
        return eval(compile(last, "<steer>", "eval"), namespace)
    return None


@register
class SteerAgent(Chare):
    """The steering chare (a singleton on PE 0): owns the CCS handler and
    runs each call in its own coroutine."""

    namespace = None            # eval globals; None: __main__'s

    def __init__(self, allow_eval, verbose, max_inflight=MAX_INFLIGHT, timeout=None):
        # a load balancer must not move it: the CCS handler and the
        # delayed-reply tokens belong to PE 0's process
        self.setMigratable(False)
        self.verbose = verbose
        self.max_inflight = max_inflight
        self.timeout = timeout
        self.calls = {}         # call id -> {"token", "replied"}, until the call returns
        self.next_id = 0
        self.auth_on, why = auth_status()
        self.tracebacks = self.auth_on or verbose
        self.eval_on = allow_eval and self.auth_on
        if allow_eval and not self.eval_on:
            # the reason (which may name the key file) goes to PE 0's stdout,
            # the client gets a generic one
            self.eval_why = ("the CCS server does not require authentication "
                             "(++server-auth with a level-0 key)")
            print("charm4py.steer: allow_eval=True ignored: %s; eval stays off "
                  "(eval is remote code execution, it needs an authenticated CCS server)"
                  % why)
        else:
            self.eval_why = "the program did not enable it (steer.init(allow_eval=True))"
        if self.namespace is None:
            self.namespace = sys.modules["__main__"].__dict__
        charm.CcsRegisterHandler(HANDLER, self.ccs_handler)

    def check(self, name, args, kwargs):
        """Raise if the request cannot run (unknown name, eval off or misused)."""
        if name == "eval":
            if not self.eval_on:
                raise PermissionError("eval is off: " + self.eval_why)
            if len(args) != 1 or not isinstance(args[0], str) or kwargs:
                raise ValueError('eval takes one argument, the source: {"call": "eval", "args": ["..."]}')
        elif name not in _registry:
            raise KeyError("no steering call '%s' (exposed: %s)"
                           % (name, ", ".join(sorted(_registry)) or "none"))

    # Runs inside the scheduler on the main thread: never blocks, always
    # answers (an exception escaping a CCS handler leaves the client waiting).
    def ccs_handler(self, data):
        try:
            name, args, kwargs = parse_request(data)
            if name == "list":
                reply = {"ok": True, "result": {"calls": describe(), "eval": self.eval_on}}
            else:
                self.check(name, args, kwargs)
                reply = {"ok": False, "error": "busy"} \
                    if len(self.calls) >= self.max_inflight else None
        except BaseException as e:
            _send(None, error_reply(e))
            if not isinstance(e, Exception):
                raise
            return
        if reply is not None:
            _send(None, reply)
            return
        try:
            self.start(name, args, kwargs)
        except Exception as e:      # only before a token was taken (start answers after)
            _send(None, error_reply(e))

    def start(self, name, args, kwargs):
        """Take a delayed-reply token and start the call; the call (or its
        timeout) sends the one reply."""
        cid = self.next_id
        self.next_id += 1
        self.calls[cid] = {"token": charm.CcsDelayReply(), "replied": False}
        try:
            if self.timeout is not None:
                charm.scheduleCallableAfter(self.timed_out, self.timeout, [cid])
            self.thisProxy.run(cid, name, args, kwargs)
        except BaseException as e:
            self.reply(cid, error_reply(e, self.tracebacks))
            self.calls.pop(cid, None)
            if not isinstance(e, Exception):
                raise

    def reply(self, cid, reply):
        """Send the one reply of call ``cid``; False (nothing sent) if it
        already has one (it timed out) or is gone."""
        call = self.calls.get(cid)
        if call is None or call["replied"]:
            return False
        call["replied"] = True
        _send(call["token"], reply)
        return True

    def timed_out(self, cid):
        """Timer callback (scheduler, main thread): answer a call still
        running with a timeout error; its own result is dropped later. A
        timer cannot be cancelled, so it may fire after the call is done."""
        self.reply(cid, {"ok": False, "error": "timeout: no result after %g s (the call keeps "
                         "running on PE 0; its result will be dropped)" % self.timeout})

    @coro
    def run(self, cid, name, args, kwargs):
        reply = {"ok": False, "error": "the call ended without a result"}
        try:
            if name == "eval":
                src = args[0]
                print("charm4py.steer: eval %r" % (src if self.verbose else src[:200]))
                result = run_source(src, self.namespace)
            else:
                if self.verbose:
                    print("charm4py.steer: %s(*%r, **%r)" % (name, args, kwargs))
                result = _registry[name](*args, **kwargs)
            if _future_types and isinstance(result, _future_types):
                result = result.get()
            reply = {"ok": True, "result": result}
        except Exception as e:
            reply = error_reply(e, self.tracebacks)
        except BaseException as e:      # SystemExit, KeyboardInterrupt, GeneratorExit
            reply = error_reply(e, self.tracebacks)
            raise                       # after the finally below has replied
        finally:
            try:
                if not self.reply(cid, reply) and self.verbose:
                    print("charm4py.steer: %s finished after its timeout; result dropped" % name)
            finally:
                self.calls.pop(cid, None)
