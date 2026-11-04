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

Return values are converted to JSON: NumPy scalars and arrays become
numbers and lists, tuples become lists, sets sorted lists, bytes text,
anything else that JSON cannot represent its ``repr``.

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
entry method would. ``list`` and malformed requests are answered at once
by the handler.

Callables are looked up in the process that runs PE 0: expose them at
module level, or from code that runs on PE 0 (``main``).

Security
--------
CCS with ``++server-auth KEYFILE`` authenticates every request: the client
proves it holds the key (SHA-1 challenge/response with a per-client
counter), and the server proves it back in each reply. It does NOT encrypt:
requests and replies, including steering arguments and results, cross the
network in the clear. Without ``++server-auth``, anyone who can reach the
CCS port can call every exposed function. Keep the port behind an ssh
forward or a private network, use ``++server-auth`` with a mode-600 key
file, and expose only what a key holder may do.

``eval`` is remote code execution with the program's privileges for
anyone who has the key. It is off unless BOTH the program calls
``steer.init(allow_eval=True)`` AND the job was started with
``++server-auth`` (checked on PE 0's command line). With
``allow_eval=True`` and no ``++server-auth``, ``init`` prints a warning
and eval stays off. Every eval request is printed on PE 0's stdout.
(When charmrun runs the CCS server, as with ``python -m charmrun.start
... ++local``, it may take ``++server-auth`` off the program's command
line; eval then stays off even though the server authenticates.)
"""

import ast
import inspect
import json
import sys
import traceback

from .charm import charm, Chare, register
from .entry_method import coro


HANDLER = "c4pSteer"
RESERVED = ("list", "eval")

_registry = {}          # name -> callable (this process)
_agent = None           # proxy of the steering chare, set by init()
_future_types = ()      # charm4py Future classes, filled in by init()


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


# -- setup -----------------------------------------------------------------------

def auth_enabled(argv=None):
    """True if the command line starts the CCS server with ++server-auth."""
    argv = sys.argv if argv is None else argv
    return any(a == "++server-auth" or a.startswith("++server-auth=") for a in argv)


def init(allow_eval=False, namespace=None, verbose=False):
    """Start serving steering requests: creates the steering chare on PE 0,
    which registers the CCS handler ``c4pSteer``. Call once, after
    ``charm.start`` (e.g. in ``main``); later calls return the same proxy.

    allow_eval: also accept ``eval`` requests, but only if the job runs with
        ``++server-auth`` (eval is remote code execution; see the module doc).
    namespace: the globals dict ``eval`` runs in (default: ``__main__``'s).
        Only used on PE 0's process; pass it from code running there.
    verbose: print one line on PE 0 per steering call (eval calls are
        always printed).
    """
    global _agent, _future_types
    if _agent is not None:
        return _agent
    from .threads import Future, CollectiveFuture
    _future_types = (Future, CollectiveFuture)
    if namespace is not None:
        SteerAgent.namespace = namespace
    _agent = Chare(SteerAgent, args=[bool(allow_eval), bool(verbose)], onPE=0)
    return _agent


# -- request handling (pure Python, no Charm++ calls) ---------------------------

def _jsonable(x):
    """json.dumps default=: NumPy, sets, bytes, then repr."""
    for conv in ("tolist", "item"):                 # numpy arrays and scalars
        f = getattr(x, conv, None)
        if callable(f):
            try:
                return f()
            except Exception:
                pass
    if isinstance(x, (set, frozenset)):
        return sorted(x, key=repr)
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", "replace")
    return repr(x)


def encode(reply):
    """A reply dict as the bytes sent to the client."""
    try:
        return json.dumps(reply, default=_jsonable).encode("utf-8")
    except Exception as e:      # e.g. a circular structure
        return json.dumps({"ok": False, "error": "result is not JSON-serializable: %s: %s"
                           % (type(e).__name__, e)}).encode("utf-8")


def error_reply(exc, with_traceback=True):
    r = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    if with_traceback and exc.__traceback__ is not None:
        r["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return r


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

    def __init__(self, allow_eval, verbose):
        self.verbose = verbose
        self.eval_on = allow_eval and auth_enabled()
        if allow_eval and not self.eval_on:
            self.eval_why = "the job was started without ++server-auth"
            print("charm4py.steer: allow_eval=True ignored: %s; eval stays off "
                  "(eval is remote code execution, it needs an authenticated CCS server)"
                  % self.eval_why)
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
                charm.CcsSendReply(encode({"ok": True, "result": {
                    "calls": describe(), "eval": self.eval_on}}))
                return
            self.check(name, args, kwargs)
        except Exception as e:
            charm.CcsSendReply(encode(error_reply(e, with_traceback=False)))
            return
        token = charm.CcsDelayReply()
        try:
            self.thisProxy.run(token, name, args, kwargs)
        except Exception as e:
            charm.CcsSendDelayedReply(token, encode(error_reply(e)))

    @coro
    def run(self, token, name, args, kwargs):
        try:
            if name == "eval":
                print("charm4py.steer: eval %r" % args[0][:200])
                result = run_source(args[0], self.namespace)
            else:
                if self.verbose:
                    print("charm4py.steer: %s(*%r, **%r)" % (name, args, kwargs))
                result = _registry[name](*args, **kwargs)
            if _future_types and isinstance(result, _future_types):
                result = result.get()
            reply = {"ok": True, "result": result}
        except Exception as e:
            reply = error_reply(e)
        charm.CcsSendDelayedReply(token, encode(reply))
