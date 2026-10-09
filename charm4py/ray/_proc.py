"""Worker processes: one Python process per actor, owned by the actor's Charm
host, so that actor code that blocks cannot stall a PE, ray.kill really ends it,
and environment variables and accelerator visibility are per actor.

The host never blocks on a pipe. Every PE polls the pipes of its workers from a
Charm timer callback: often while a request is outstanding and a reply was seen
recently, less often while a long call runs, and rarely when idle (which still
notices a worker that dies)."""

import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Connection, wait as _wait_ready

FAST = 0.0002   # a worker spoke in the last _RECENT seconds
SLOW = 0.002    # a request is outstanding, or a worker spoke in the last _ACTIVE s
IDLE = 0.01     # quiet; still notices a request from a worker within 10 ms
_RECENT = 0.05
_ACTIVE = 1.0

_workers = {}           # Connection -> WorkerProcess
_exited = []            # killed processes not yet reaped
_generation = 0         # only the newest scheduled poll keeps polling
_poll_due = None        # when that poll runs (monotonic), None if none is scheduled
_last_message = 0.0


def _charm():
    import charm4py
    return charm4py.charm


class WorkerProcess(object):
    """A worker process started by this PE, or (``sock``) a process that
    connected to it (see Listener)."""

    def __init__(self, env, on_message, on_exit, sock=None, pid=None):
        self.on_message = on_message
        self.on_exit = on_exit
        self.outstanding = 0
        self.closed = False
        if sock is not None:
            self.proc = None
            self._pid = pid
            self.conn = Connection(sock.detach())
            _workers[self.conn] = self
            _ensure_polling()
            return
        parent_sock, child_sock = socket.socketpair()
        full_env = dict(os.environ)
        full_env.update(env)
        full_env['CHARM_NOLOAD'] = '1'          # the worker does not load Charm++
        full_env['CHARM4PY_RAY_WORKER'] = '1'
        cmd = [sys.executable, '-m', 'charm4py.ray._worker_main', str(child_sock.fileno())]
        self.proc = subprocess.Popen(cmd, env=full_env, pass_fds=(child_sock.fileno(),))
        child_sock.close()
        self._pid = self.proc.pid
        self.conn = Connection(parent_sock.detach())
        _workers[self.conn] = self
        _ensure_polling()

    @property
    def pid(self):
        return self._pid

    def send(self, data, request=False):
        """Sends one message (bytes). ``request``: a reply is expected, so poll
        soon. A dead worker is noticed by the poller."""
        if self.closed:
            return False
        try:
            self.conn.send_bytes(data)
        except OSError:
            return False
        if request:
            self.outstanding += 1
            _ensure_polling(urgent=True)
        return True

    def kill(self):
        """Terminates the process now (a connected process is only
        disconnected). on_exit is not called."""
        if self.closed:
            return
        self._detach()
        if self.proc is None:
            return
        try:
            self.proc.kill()
        except OSError:
            pass
        _exited.append(self.proc)
        _ensure_polling()

    def _detach(self):
        self.closed = True
        _workers.pop(self.conn, None)
        try:
            self.conn.close()
        except OSError:
            pass

    def _lost(self):
        """The pipe closed: the process exited or crashed."""
        if self.closed:
            return
        self._detach()
        if self.proc is None:
            self.on_exit(None)
            return
        try:
            code = self.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        self.on_exit(code)


class Listener(object):
    """A Unix socket that other processes of the job connect to (ray.init in a
    process this job did not start, e.g. one started by the driver).
    ``on_connect(sock)`` gets each accepted connection."""

    def __init__(self, on_connect):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix='c4pray-')   # private (mode 0700)
        self.path = os.path.join(self.dir, 'sock')
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(64)
        self.sock.setblocking(False)
        self.on_connect = on_connect
        _listeners[self.sock] = self
        _ensure_polling()

    def _accept(self):
        while True:
            try:
                sock, _ = self.sock.accept()
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            sock.setblocking(True)
            self.on_connect(sock)

    def close(self):
        _listeners.pop(self.sock, None)
        try:
            self.sock.close()
            os.unlink(self.path)
            os.rmdir(self.dir)
        except OSError:
            pass


_listeners = {}         # socket -> Listener


def _poll(generation):
    global _poll_due, _last_message
    if generation != _generation:
        return   # superseded by an earlier poll scheduled for an urgent request
    _poll_due = None
    if _workers or _listeners:
        for conn in _wait_ready(list(_workers) + list(_listeners), timeout=0):
            listener = _listeners.get(conn)
            if listener is not None:
                listener._accept()
                continue
            worker = _workers.get(conn)
            while worker is not None and not worker.closed:
                try:
                    if not conn.poll(0):
                        break
                    data = conn.recv_bytes()
                except (EOFError, OSError):
                    worker._lost()
                    break
                _last_message = time.monotonic()
                worker.on_message(data)
    for proc in list(_exited):
        if proc.poll() is not None:
            _exited.remove(proc)
    _ensure_polling()


def _ensure_polling(urgent=False):
    """Schedules the next poll. ``urgent`` (a request was just sent) moves an
    idle-rate poll forward."""
    global _generation, _poll_due
    if not (_workers or _exited or _listeners):
        return
    quiet = time.monotonic() - _last_message
    if urgent or quiet < _RECENT:
        interval = FAST
    elif quiet < _ACTIVE or any(w.outstanding for w in _workers.values()):
        interval = SLOW
    else:
        interval = IDLE
    due = time.monotonic() + interval
    if _poll_due is not None and _poll_due <= due:
        return
    _generation += 1
    _poll_due = due
    _charm().scheduleCallableAfter(_poll, interval, [_generation])
