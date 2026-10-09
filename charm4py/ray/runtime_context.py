"""``ray.get_runtime_context()``."""

import json

from . import api
from ._options import ACCELERATOR_KEYS


class RuntimeContext(object):
    """Information about the driver, task or actor that is currently running."""

    def _ctx(self):
        return api._current_context()

    def get_node_id(self):
        ctx = self._ctx()
        if ctx is not None and ctx.node_id is not None:
            return ctx.node_id
        api._ensure_initialized()
        return api._my_node()['NodeID']

    def get_job_id(self):
        return api._job_id()

    def get_worker_id(self):
        return api._worker_id()

    def get_task_id(self):
        ctx = self._ctx()
        return ctx.task_id if ctx is not None else None

    def get_actor_id(self):
        ctx = self._ctx()
        return ctx.actor_id if ctx is not None else None

    def get_actor_name(self):
        ctx = self._ctx()
        return ctx.actor_name if ctx is not None else None

    def get_placement_group_id(self):
        ctx = self._ctx()
        return ctx.pg_id if ctx is not None else None

    def get_accelerator_ids(self):
        ctx = self._ctx()
        assigned = ctx.accelerator_ids if ctx is not None else {}
        return {key: list(assigned.get(key, [])) for key in ACCELERATOR_KEYS}

    def get_assigned_resources(self):
        ctx = self._ctx()
        return dict(ctx.assigned_resources) if ctx is not None else {}

    def get_node_labels(self):
        return {'ray.io/node_id': self.get_node_id()}

    @property
    def namespace(self):
        return api._namespace()

    @property
    def runtime_env(self):
        ctx = self._ctx()
        if ctx is not None and ctx.runtime_env is not None:
            return dict(ctx.runtime_env)
        return dict(api._current_job().get('runtime_env') or {})

    def get_runtime_env_string(self):
        return json.dumps(self.runtime_env)

    @property
    def was_current_actor_reconstructed(self):
        return False

    @property
    def gcs_address(self):
        """Where other processes connect to this job: ray.init(address=...) or
        RAY_ADDRESS."""
        return api._address or api.get_node_ip_address() + ':0'

    def get(self):
        out = {'job_id': self.get_job_id(), 'node_id': self.get_node_id(),
               'namespace': self.namespace}
        if self.get_task_id() is not None:
            out['task_id'] = self.get_task_id()
        if self.get_actor_id() is not None:
            out['actor_id'] = self.get_actor_id()
        return out


def get_runtime_context():
    return RuntimeContext()
