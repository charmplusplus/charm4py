"""Entry point of a worker process (see _proc.py).

    python -m charm4py.ray._worker_main FD

FD is this end of a socket pair with the worker's host chare. A worker does not
load Charm++: its ``ray`` API forwards to the host (api._client). It is one of

- an actor worker: constructs one actor and runs its calls one at a time;
- a task worker: runs tasks one at a time, for a pool kept by a PE;
- the driver: runs the user's script (charm4py.ray.launch).

A process that connects to the job (api._connect) uses the same _Channel.

Messages from the host:  ('init', info) ('construct', rid, desc, args, kwargs)
                         ('call', rid, method, args, kwargs, task_id)
                         ('task', rid, desc, args, kwargs, context, name)
                         ('api_reply', rid, ok, value)
Messages to the host:    ('done', rid, status, value)  status: ok | error | exit
                         ('api', rid, name, args, kwargs, job)
                         ('driver_exit', code)

A reader thread receives everything: API replies go to the thread that made the
call (any thread of the worker may use ray), work goes to the main thread."""

import os
import queue
import sys
import threading
import traceback
from multiprocessing.connection import Connection


class _Channel(object):
    """The connection to the host, shared by every thread of the worker."""

    def __init__(self, conn, side):
        from charm4py.ray import api
        self.conn = conn
        self.side = side
        self.current_job = api._current_job
        self.send_lock = threading.Lock()
        self.work = queue.Queue()
        self.calls = {}               # rid -> [event, reply]
        self.calls_lock = threading.Lock()
        self.next_rid = 0
        self.closed = False
        threading.Thread(target=self._read, name='charm4py-ray-reader', daemon=True).start()

    def send(self, msg):
        data = self.side.dumps(msg)
        with self.send_lock:
            self.conn.send_bytes(data)

    def _read(self):
        try:
            while True:
                msg = self.side.loads(self.conn.recv_bytes())
                if msg[0] == 'api_reply':
                    with self.calls_lock:
                        slot = self.calls.pop(msg[1], None)
                    if slot is not None:
                        slot[1] = msg
                        slot[0].set()
                else:
                    self.work.put(msg)
        except (EOFError, OSError):
            pass
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
        self.send(('api', rid, api_name, args, kwargs, self.current_job()))
        slot[0].wait()
        reply = slot[1]
        if reply is None:
            raise _host_gone()
        if reply[2]:
            return reply[3]
        raise reply[3]


    def close(self):
        try:
            self.conn.close()
        except OSError:
            pass


def _host_gone():
    from charm4py.ray.exceptions import RaySystemError
    return RaySystemError('the Charm4py process that owns this worker has exited')


def _reply(channel, rid, status, value):
    try:
        channel.send(('done', rid, status, value))
    except Exception as e:
        # the result could not be serialized
        from charm4py.ray import _errors
        channel.send(('done', rid, 'error', _errors.task_error('<result>', e)))


def _run_actor(channel, info, api):
    from charm4py.ray import _errors
    from charm4py.ray._serialization import resolve
    from charm4py.ray.actor import _ActorExit
    ctx = api._ExecContext(**info['context'])
    api._job = ctx.job
    instance = None
    while True:
        msg = channel.work.get()
        if msg is None:
            return 0
        kind = msg[0]
        if kind == 'construct':
            _, rid, desc, args, kwargs = msg
            ctx.task_id = api._new_task_id()
            api._set_context(ctx)
            try:
                instance = resolve(desc)(*args, **kwargs)
                _reply(channel, rid, 'ok', None)
            except Exception as e:
                _reply(channel, rid, 'error', _errors.task_error(
                    info['class_name'] + '.__init__', e, actor_id=ctx.actor_id))
        elif kind == 'call':
            _, rid, method, args, kwargs, task_id = msg
            ctx.task_id = task_id
            api._set_context(ctx)
            try:
                _reply(channel, rid, 'ok', getattr(instance, method)(*args, **kwargs))
            except _ActorExit:
                _reply(channel, rid, 'exit', None)
                return 0
            except SystemExit as e:
                _reply(channel, rid, 'exit', e.code)
                return 0
            except Exception as e:
                try:
                    actor_repr = repr(instance)
                except Exception:
                    actor_repr = None
                _reply(channel, rid, 'error', _errors.task_error(
                    info['class_name'] + '.' + method, e, actor_repr=actor_repr,
                    actor_id=ctx.actor_id))
        else:
            raise RuntimeError('unexpected message for an actor worker: ' + repr(kind))


def _run_tasks(channel, info, api):
    from charm4py.ray import _errors
    from charm4py.ray._serialization import resolve
    while True:
        msg = channel.work.get()
        if msg is None:
            return 0
        _, rid, desc, args, kwargs, context, task_name = msg
        ctx = api._ExecContext(**context)
        api._set_context(ctx)
        api._job = ctx.job      # also for threads the task starts
        name = task_name or (desc[2] if desc[0] == 'ref' else 'task')
        try:
            fn = resolve(desc)
            name = task_name or getattr(fn, '__qualname__', name)
            _reply(channel, rid, 'ok', fn(*args, **kwargs))
        except SystemExit as e:
            _reply(channel, rid, 'error', _errors.crash_error(name, e.code))
        except Exception as e:
            _reply(channel, rid, 'error', _errors.task_error(name, e))


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
        channel.send(('driver_exit', code))
    except OSError:
        pass
    return code


def _main():
    from charm4py.ray import _ipc, api, object_ref
    object_ref.INERT = True
    conn = Connection(int(sys.argv[1]))
    side = _ipc.WorkerSide()
    msg = side.loads(conn.recv_bytes())
    assert msg[0] == 'init', msg
    info = msg[1]
    for path in reversed(info['sys_path']):
        if path not in sys.path:
            sys.path.insert(0, path)
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
