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
* ``steer.init(allow_eval=False, namespace=None, verbose=False)``: call
  once after ``charm.start`` (in ``main``); creates the steering chare on
  PE 0, which registers the handler. Import ``charm4py.steer`` before
  ``charm.start`` so its chare type is registered on every process.

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
(see Security). Results are converted to JSON; NumPy values become numbers
and lists, other objects JSON cannot hold their ``repr``.

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

Security
--------

``++server-auth`` authenticates each request (SHA-1 challenge/response
with a per-client counter) and each reply; it does not encrypt. Without
it anyone who can reach the CCS port can call every exposed function.
Keep the port private (an ssh tunnel, the cluster network), keep the key
file mode 600, and expose only what a key holder may do.

``eval`` is remote code execution for anyone with the key. It is off
unless the program calls ``steer.init(allow_eval=True)`` and the job runs
with ``++server-auth`` (checked on PE 0's command line); with
``allow_eval=True`` and no auth, ``init`` prints a warning and eval stays
off. Each eval is printed on PE 0's stdout.
