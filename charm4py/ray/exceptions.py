"""Exceptions with the names, class hierarchy and messages of ``ray.exceptions``
(Ray 2.56). Only the exceptions that the Charm4py implementation can raise, or
that applications commonly catch, are provided."""

import os


class RayError(Exception):
    """Super class of all ray exception types."""


class RaySystemError(RayError):
    pass


class RayChannelError(RaySystemError):
    """A system error of Ray's compiled-graph channels (not implemented by
    charm4py.ray; defined for code that catches it)."""


class RayChannelTimeoutError(RayChannelError, TimeoutError):
    pass


class TaskCancelledError(RayError):
    pass


class GetTimeoutError(RayError, TimeoutError):
    """Indicates that a call to the worker timed out."""


class ObjectLostError(RayError):
    pass


class OwnerDiedError(ObjectLostError):
    pass


class ActorUnavailableError(RayError):
    pass


class ActorAlreadyExistsError(ValueError, RayError):
    pass


class RayActorError(RayError):
    """Indicates that the actor died unexpectedly before finishing a task."""

    def __init__(self, cause=None, actor_id=None, error_msg=None, actor_init_failed=False):
        self.actor_id = actor_id
        self.actor_init_failed = actor_init_failed
        self.error_msg = error_msg or 'The actor died unexpectedly before finishing this task.'
        self.cause = cause
        super().__init__(self.error_msg)

    def __reduce__(self):
        return (type(self), (self.cause, self.actor_id, self.error_msg, self.actor_init_failed))

    def __str__(self):
        return self.error_msg


class ActorDiedError(RayActorError):
    """Indicates that the actor died, either because it was killed, it exited, or
    its creation task failed."""


class RayTaskError(RayError):
    """Indicates that a task threw an exception during execution.

    ``ray.get`` raises the result of :meth:`as_instanceof_cause`, which is also an
    instance of the exception class the task raised, so applications can catch the
    original exception type."""

    def __init__(self, function_name, traceback_str, cause, proctitle=None, pid=None,
                 ip=None, actor_repr=None, actor_id=None):
        self.function_name = function_name
        self.traceback_str = traceback_str
        self.cause = cause
        self.proctitle = proctitle or ('ray::' + str(function_name))
        self.pid = os.getpid() if pid is None else pid
        self.ip = ip
        self.actor_repr = actor_repr
        self.actor_id = actor_id
        super().__init__(self._message())

    def __reduce__(self):
        return (RayTaskError, (self.function_name, self.traceback_str, self.cause,
                               self.proctitle, self.pid, self.ip, self.actor_repr,
                               self.actor_id))

    def _message(self):
        out = [self.proctitle + '() (pid=' + str(self.pid) + ', ip=' + str(self.ip)]
        if self.actor_id is not None:
            out[0] += ', actor_id=' + str(self.actor_id)
        if self.actor_repr is not None:
            out[0] += ', repr=' + str(self.actor_repr)
        out[0] += ')'
        if self.traceback_str:
            out.append(self.traceback_str.rstrip('\n'))
        return '\n'.join(out)

    def __str__(self):
        return self._message()

    def as_instanceof_cause(self):
        """Returns an exception that is an instance of both RayTaskError and the
        class of the exception that caused it, so ``except ValueError`` catches a
        ``ValueError`` raised by a remote task."""
        cause_cls = type(self.cause)
        if self.cause is None or issubclass(cause_cls, RayTaskError) or \
                not issubclass(cause_cls, Exception):
            return self
        error_msg = str(self)
        cause = self.cause

        class cls(RayTaskError, cause_cls):
            def __init__(self, cause):
                self.cause = cause
                # BaseException implements __new__ and stores the args there, so
                # set them directly rather than calling the cause's __init__
                self.args = (cause,)

            def __getattr__(self, name):
                return getattr(self.cause, name)

            def __str__(self):
                return error_msg

        name = 'RayTaskError(' + cause_cls.__name__ + ')'
        cls.__name__ = name
        cls.__qualname__ = name
        try:
            err = cls(cause)
        except TypeError:
            # the cause's class has an incompatible instance layout
            return self
        err.function_name = self.function_name
        err.traceback_str = self.traceback_str
        err.proctitle = self.proctitle
        err.pid = self.pid
        err.ip = self.ip
        err.actor_repr = self.actor_repr
        err.actor_id = self.actor_id
        return err
