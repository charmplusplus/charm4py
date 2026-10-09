"""Entry point of a worker process (see _proc.py).

    python -m charm4py.ray._worker_main FD

FD is this end of a socket pair with the worker's host chare. A worker does not
load Charm++: its ``ray`` API forwards to the host (api._client). It is one of

- an actor worker: constructs one actor and runs its calls one at a time;
- a task worker: runs tasks one at a time, for a pool kept by a PE;
- the driver: runs the user's script (charm4py.ray.launch).

A process that connects to the job (api._connect) uses the same _Channel.

Each message is a header ``(kind, rid)`` and a payload (see _ipc.py):

  from the host   init        info
                  construct   (class desc, args, kwargs)
                  call        (method, args, kwargs, task id, num_returns)
                  task        (function desc, args, kwargs, context, name, num_returns)
                  api_reply   (ok, value)
  to the host     done        (status, value)   ok: [Opaque per return] | error: Opaque
                                                exit: exit code
                  api         (name, args, kwargs, job)
                  driver_exit code

Arguments arrive resolved: a top-level ObjectRef argument becomes the stored
value (an Opaque, unpickled here, or a wrapper for None or an error).

A reader thread receives everything: API replies go to the thread that made the
call (any thread of the worker may use ray), work goes to the main thread, which
deserializes it -- so importing user modules, and failing to, happens there."""

import os
import queue
import sys
import threading
import traceback
from multiprocessing.connection import Connection


class _Channel(object):
    """The connection to the host, shared by every thread of the worker."""

    def __init__(self, conn, side):
        from charm4py.ray import api, _ipc
        self.conn = conn
        self.side = side
        self.ipc = _ipc
        self.current_job = api._current_job
        self.send_lock = threading.Lock()
        self.work = queue.Queue()     # (kind, rid, payload bytes) for the main thread
        self.calls = {}               # rid -> [event, (ok, value)]
        self.calls_lock = threading.Lock()
        self.next_rid = 0
        self.closed = False
        threading.Thread(target=self._read, name='charm4py-ray-reader', daemon=True).start()

    def send(self, kind, rid, payload):
        data = self.side.dumps(payload)
        header = self.ipc.encode_header(kind, rid)
        with self.send_lock:
            self.conn.send_bytes(header)
            self.conn.send_bytes(data)

    def _read(self):
        try:
            while True:
                kind, rid = self.ipc.decode_header(self.conn.recv_bytes())
                data = self.conn.recv_bytes()
                if kind == 'api_reply':
                    try:
                        reply = self.side.loads(data)
                    except BaseException as e:
                        reply = (False, _undecodable('the reply of a ray call', e))
                    with self.calls_lock:
                        slot = self.calls.pop(rid, None)
                    if slot is not None:
                        slot[1] = reply
                        slot[0].set()
                else:
                    self.work.put((kind, rid, data))
        except (EOFError, OSError):
            pass
        except BaseException:
            traceback.print_exc()
        self.closed = True
        self.work.put(None)
        with self.calls_lock:
            slots, self.calls = list(self.calls.values()), {}
        for slot in slots:
            slot[0].set()

    def call(self, api_name, /, *args, **kwargs):
        """Runs ray.<api_name> on the host (api._client)."""
        slot = [threading.Event(), None]
        with self.calls_lock:
            if self.closed:
                raise _host_gone()
            rid = self.next_rid
            self.next_rid += 1
            self.calls[rid] = slot
        self.send('api', rid, (api_name, args, kwargs, self.current_job()))
        slot[0].wait()
        reply = slot[1]
        if reply is None:
            raise _host_gone()
        if reply[0]:
            return reply[1]
        raise reply[1]

    def opaque(self, value):
        return self.side.opaque(value)

    def close(self):
        try:
            self.conn.close()
        except OSError:
            pass


def _host_gone():
    from charm4py.ray.exceptions import RaySystemError
    return RaySystemError('the Charm4py process that owns this worker has exited')


def _undecodable(what, error):
    from charm4py.ray.exceptions import RaySystemError
    return RaySystemError(what + ' could not be deserialized in this worker process: ' +
                          type(error).__name__ + ': ' + str(error))


def _reply_ok(channel, rid, values, function_name):
    try:
        encoded = [channel.opaque(v) for v in values]
    except Exception as e:
        # a result could not be serialized
        from charm4py.ray import _errors
        _reply_error(channel, rid, _errors.task_error(function_name, e))
        return
    channel.send('done', rid, ('ok', encoded))


def _reply_error(channel, rid, error):
    try:
        encoded = channel.opaque(error)
    except Exception as e:
        from charm4py.ray.exceptions import RaySystemError
        encoded = channel.opaque(RaySystemError(
            'the error of this task could not be serialized: ' + repr(e) + '\n' +
            str(error)))
    channel.send('done', rid, ('error', encoded))


def _payload(channel, data, rid, what):
    """The deserialized payload, or None after replying with the error."""
    try:
        return channel.side.loads(data)
    except BaseException as e:
        traceback.print_exc()
        _reply_error(channel, rid, _undecodable(what, e))
        return None


def _run_actor(channel, info, api):
    from charm4py.ray import _errors
    from charm4py.ray._serialization import resolve
    from charm4py.ray.actor import _ActorExit
    ctx = api._ExecContext(**info['context'])
    api._job = ctx.job
    class_name = info['class_name']
    instance = None
    while True:
        msg = channel.work.get()
        if msg is None:
            return 0
        kind, rid, data = msg
        if kind == 'construct':
            payload = _payload(channel, data, rid, 'the actor class or its arguments')
            if payload is None:
                continue
            desc, args, kwargs = payload
            ctx.task_id = api._new_task_id()
            api._set_context(ctx)
            name = class_name + '.__init__'
            try:
                args, kwargs = api._unwrap_args(args, kwargs)
                instance = resolve(desc)(*args, **kwargs)
                channel.send('done', rid, ('ok', None))
            except api._DependencyFailed as e:
                _reply_error(channel, rid, api._creation_error(
                    api._dependency_error(name, e.error), ctx.actor_id))
            except Exception as e:
                _reply_error(channel, rid, api._creation_error(
                    _errors.task_error(name, e, actor_id=ctx.actor_id), ctx.actor_id))
        elif kind == 'call':
            payload = _payload(channel, data, rid, 'the arguments of an actor method')
            if payload is None:
                continue
            method, args, kwargs, task_id, num_returns = payload
            ctx.task_id = task_id
            api._set_context(ctx)
            name = class_name + '.' + method
            try:
                args, kwargs = api._unwrap_args(args, kwargs)
            except api._DependencyFailed as e:
                _reply_error(channel, rid, api._dependency_error(name, e.error))
                continue
            try:
                result = getattr(instance, method)(*args, **kwargs)
            except _ActorExit:
                channel.send('done', rid, ('exit', None))
                return 0
            except SystemExit as e:
                channel.send('done', rid, ('exit', e.code))
                return 0
            except Exception as e:
                try:
                    actor_repr = repr(instance)
                except Exception:
                    actor_repr = None
                _reply_error(channel, rid, _errors.task_error(
                    name, e, actor_repr=actor_repr, actor_id=ctx.actor_id))
                continue
            try:
                values = api._split_results(result, num_returns, name)
            except api._ResultsMismatch as e:
                _reply_error(channel, rid, e.error)
                continue
            _reply_ok(channel, rid, values, name)
        else:
            raise RuntimeError('unexpected message for an actor worker: ' + repr(kind))


def _run_tasks(channel, info, api):
    from charm4py.ray import _errors
    from charm4py.ray._serialization import resolve
    while True:
        msg = channel.work.get()
        if msg is None:
            return 0
        kind, rid, data = msg
        payload = _payload(channel, data, rid, 'the task function or its arguments')
        if payload is None:
            continue
        desc, args, kwargs, context, task_name, num_returns = payload
        ctx = api._ExecContext(**context)
        api._set_context(ctx)
        api._job = ctx.job      # also for threads the task starts
        name = task_name or (desc[2] if desc[0] == 'ref' else 'task')
        try:
            fn = resolve(desc)
            name = task_name or getattr(fn, '__qualname__', name)
            args, kwargs = api._unwrap_args(args, kwargs)
            result = fn(*args, **kwargs)
            values = api._split_results(result, num_returns, name)
        except api._DependencyFailed as e:
            _reply_error(channel, rid, api._dependency_error(name, e.error))
            continue
        except api._ResultsMismatch as e:
            _reply_error(channel, rid, e.error)
            continue
        except SystemExit as e:
            _reply_error(channel, rid, _errors.crash_error(name, e.code))
            continue
        except Exception as e:
            _reply_error(channel, rid, _errors.task_error(name, e))
            continue
        _reply_ok(channel, rid, values, name)


def _run_driver(channel, info, api):
    from charm4py.ray.launch import _load_script
    sys.argv = list(info['argv'])
    code = 0
    try:
        _load_script(info['main_script'], '__main__')
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = 1
    # what the interpreter does at exit -- wait for non-daemon threads, then run
    # the atexit handlers (applications stop their own processes there) -- while
    # Ray still works; then end the job, like Ray's own atexit handler
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and not thread.daemon:
            thread.join()
    try:
        import atexit
        atexit._run_exitfuncs()
    except BaseException:
        traceback.print_exc()
    try:
        if api.is_initialized():
            api.shutdown()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        channel.send('driver_exit', None, code)
    except OSError:
        pass
    return code


def _main():
    from charm4py.ray import _ipc, api, object_ref
    object_ref.INERT = True
    conn = Connection(int(sys.argv[1]))
    side = _ipc.WorkerSide()
    kind, _ = _ipc.decode_header(conn.recv_bytes())
    assert kind == 'init', kind
    info = side.loads(conn.recv_bytes())
    for path in info['sys_path']:
        # after this process's own paths, so a package directory the host
        # happened to put on its path cannot shadow installed packages
        if path not in sys.path:
            sys.path.append(path)
    from charm4py.ray.launch import install_ray_facade, _load_script
    install_ray_facade(worker=True)
    channel = _Channel(conn, side)
    api._client = channel
    api.__dict__.update(info['globals'])
    if api._address:
        # processes this one starts can connect to the job
        os.environ[api.ADDRESS_ENV] = api._address
    if info['kind'] != 'driver' and info.get('main_script'):
        _load_script(info['main_script'], '__mp_main__')
    run = {'actor': _run_actor, 'tasks': _run_tasks, 'driver': _run_driver}[info['kind']]
    return run(channel, info, api)


if __name__ == '__main__':
    try:
        code = _main()
    except KeyboardInterrupt:
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)
