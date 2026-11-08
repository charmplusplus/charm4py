===================
Steering (CCS)
===================

``charm4py.steer`` lets a client call Python functions of a running program
by name over CCS (Converse Client-Server), for example to change a
parameter of a simulation while it runs. The program exposes callables;
the module registers one CCS handler, ``c4pSteer``, on PE 0.

.. code-block:: Python

    from charm4py import charm, Chare, Array, steer

    @steer.expose
    def set_rate(rate: float):
        "Set the diffusion rate on every chare."
        cells.set_rate(rate, awaitable=True).get()   # broadcast and wait
        return rate

    def main(args):
        global cells
        cells = Array(Cell, 16)
        steer.init()
        ...

    charm.start(main)

Run with ``++server ++server-port N`` and, unless the port is private,
``++server-auth KEYFILE``. ``examples/steer/steer_demo.py`` is a complete
program with liveViz. appSpector (github.com/charmplusplus/appSpector) has
a client: ``ccsio.steer`` and the shell's ``steer`` command.

API
---

* ``steer.expose(fn, name=None)``, also as ``@steer.expose``,
  ``@steer.expose("name")``: make ``fn`` callable under ``name``
  (default ``fn.__name__``). ``list`` and ``eval`` are reserved.
  Expose at module level or from code running on PE 0.
* ``steer.unexpose(name)``, ``steer.exposed()``.
* ``steer.init(allow_eval=False, namespace=None, verbose=False,
  max_inflight=64, timeout=None)``: call once after ``charm.start``, in PE
  0's process (in ``main``); creates the steering chare on PE 0, which
  registers the handler. Import ``charm4py.steer`` before ``charm.start``
  so its chare type is registered on every process. Exposures must happen
  in PE 0's process too (module level or ``main``): the handler only sees
  PE 0's registry. ``verbose`` prints each call with its arguments and
  sends tracebacks even without authentication; ``max_inflight`` bounds
  the calls running at once; ``timeout`` (seconds) answers a call that
  runs longer with a timeout error.

Request and reply
-----------------

The request is UTF-8 JSON ``{"call": NAME, "args": [...], "kwargs": {...}}``,
sent to PE 0. The reply is UTF-8 JSON, exactly one per request:
``{"ok": true, "result": VALUE}`` or
``{"ok": false, "error": "Type: message", "traceback": "..."}``.
``{"call": "list"}`` answers
``{"calls": [{"name", "signature", "doc"}, ...], "eval": BOOL}``.
``{"call": "eval", "args": [SOURCE]}`` runs Python source on PE 0 and
answers the value of an expression or of a trailing expression statement
(see Security). Results are converted to strict JSON: NumPy arrays and
scalars (real ``numpy.ndarray`` / ``numpy.generic`` only) become lists and
numbers, dict keys become strings (``str(key)``), NaN and +-Infinity become
``null``, other objects JSON cannot hold their ``repr``. A traceback is
included only when the server authenticates (see Security) or with
``verbose=True``.

Execution
---------

The CCS handler runs on PE 0 in the scheduler, where a future cannot be
waited on. It parses the request, takes a delayed-reply token
(``CcsDelayReply``) and passes the call to a coroutine entry method of the
steering chare, which runs the callable and answers with
``CcsSendDelayedReply``. A callable may block on futures
(``awaitable=True``, ``ret=True``, ``charm.sleep``); while it waits, PE 0
keeps scheduling other work, including other CCS requests such as liveViz
images. A callable that returns a Future gets that future's value sent.
Computation that never yields holds PE 0, as in any entry method.

At most ``max_inflight`` calls (default 64) run at once; another request
is answered ``{"ok": false, "error": "busy"}`` at once. With ``timeout``, a
call still running after that many seconds is answered with a timeout
error; it cannot be stopped, keeps its slot until it returns, and its
result is dropped. Every request gets exactly one reply, also when the
callable raises ``SystemExit``, ``KeyboardInterrupt`` or ``GeneratorExit``
(the reply is sent first, then the exception propagates) or when the
result cannot be encoded or sent (a minimal error reply is tried).

Security
--------

``++server-auth`` authenticates each request (SHA-1 challenge/response
with a per-client counter) and each reply; it does not encrypt. Without
it anyone who can reach the CCS port can call every exposed function.
Keep the port private (an ssh tunnel, the cluster network), keep the key
file mode 600, and expose only what a key holder may do.

Charm++'s CCS server (``CcsSecMan_default`` in ``conv-ccs/ccs-server.C``;
Reconverse has the same code) reads the key file as ``LEVEL KEY`` lines and
lets UNAUTHENTICATED requests through whenever the file has no level-0 key,
even with ``++server-auth``: an empty file, only ``1 <hex>`` lines, or only
comments leave the port open. Always put a ``0 <hex key>`` line in it.

``eval`` is remote code execution for anyone with a key. It is on only if
all of these hold:

* the program calls ``steer.init(allow_eval=True)``;
* PE 0's command line (``sys.argv``) has ``++server-auth FILE`` as two
  arguments (the only form Charm++ accepts; ``++server-auth=FILE`` does not
  count);
* every such FILE is readable by PE 0's process (relative names against
  the working directory at import of ``charm4py.steer``) and has a level-0
  key, parsed the way Charm++ parses it: ``fgets`` pieces of at most 199
  bytes, ``sscanf("%d%s")``, level 0, and a key starting with two hex
  digits or ``OTP``.

Otherwise ``init`` prints the reason on PE 0 and eval stays off (clients
get a generic "eval is off" error that does not name the file). The
handler cannot see a request's security level (charm4py does not expose
``CcsSecAttr``): with a level-0 key every request that reaches it was
authenticated, but possibly with a key of another level in the same file,
so every key in that file can eval. When charmrun runs the CCS server it
may take ``++server-auth`` off the program's command line; eval then stays
off. Each eval is printed on PE 0's stdout (first 200 characters; all with
``verbose``). Call arguments are printed only with ``verbose``.
