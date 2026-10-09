"""Charm chares that implement Ray: _RayWorker (one per PE) runs tasks,
_ActorHost (one array element per actor) runs an actor, and _RayControl (one, on
PE 0) owns actor names, node resources and placement groups.

Registered by charm._registerInternalChares on every PE at charm.start()."""

import collections
import os
import pickle
import traceback

from greenlet import getcurrent

from charm4py import Chare, coro
from . import api
from ._options import ACCELERATOR_KEYS
from ._serialization import resolve
from .actor import _ActorExit, killed_message
from .exceptions import ActorDiedError, RayTaskError, RaySystemError

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_CHARM4PY_DIR = os.path.dirname(_PKG_DIR)


def _user_traceback(exc):
    """The traceback of ``exc`` without the frames of charm4py itself."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if not os.path.abspath(f.filename).startswith(_CHARM4PY_DIR)]
    lines = traceback.format_list(frames)
    lines += traceback.format_exception_only(type(exc), exc)
    return ''.join(lines)


def _portable_cause(exc):
    """``exc`` if it survives pickling (it is stored in the object store and may be
    sent to other processes), else None."""
    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:
        return None


def _task_error(function_name, exc, actor_repr=None, actor_id=None):
    tb = _user_traceback(exc)
    cause = _portable_cause(exc)
    if cause is None:
        cause = RaySystemError(tb)
    return RayTaskError(function_name, tb, cause, ip=api.get_node_ip_address(),
                        actor_repr=actor_repr, actor_id=actor_id)


def _crash_error(function_name, code):
    return RayTaskError(function_name, '  SystemExit: ' + str(code) +
                        '\nThe task called sys.exit()', RaySystemError(
                            'The worker exited while executing this task (sys.exit(' +
                            str(code) + '))'), ip=api.get_node_ip_address())


class _RayWorker(Chare):

    def node_info(self):
        return api._local_node_info()

    @coro
    def execute_task(self, fn_desc, args, kwargs, return_refs, name, pg_info):
        api._set_context(api._ExecContext(
            task_id=api._new_task_id(), node_id=api._my_node()['NodeID'],
            pg_id=pg_info[0] if pg_info else None,
            pg_bundle_specs=pg_info[1] if pg_info else None))
        function_name = name or (fn_desc[2] if fn_desc[0] == 'ref' else 'task')
        try:
            fn = resolve(fn_desc)
            if name is None:
                function_name = getattr(fn, '__qualname__', function_name)
            try:
                args, kwargs = api._resolve_args(args, kwargs)
            except api._DependencyFailed as e:
                err = api._dependency_error(function_name, e.error)
                for ref in return_refs:
                    api._store_error(ref, err)
                return
            result = fn(*args, **kwargs)
        except SystemExit as e:
            err = _crash_error(function_name, e.code)
            for ref in return_refs:
                api._store_error(ref, err)
            return
        except Exception as e:
            err = _task_error(function_name, e)
            for ref in return_refs:
                api._store_error(ref, err)
            return
        api._store_results(return_refs, result, function_name)


class _ActorHost(Chare):
    """Runs one actor. Method calls execute one at a time, in the order each
    caller (handle) made them, even when a method blocks in ray.get()."""

    def __init__(self, spec):
        self.spec = spec
        self.state = 'WAIT_CTOR'   # -> 'ALIVE' -> 'DEAD'
        self.instance = None
        self.death = None
        self.queue = collections.deque()
        self.expected = {}          # caller -> next sequence number to run
        self.held = {}              # caller -> {seq: call} that arrived early
        self.running = False
        self.setMigratable(False)
        self.ctx = api._ExecContext(
            actor_id=spec['actor_id'], actor_name=spec['name'],
            accelerator_ids=spec['accelerator_ids'], pg_id=spec['pg_id'],
            pg_bundle_specs=spec['pg_bundle_specs'], runtime_env=spec['runtime_env'],
            assigned_resources=spec['assigned_resources'], node_id=spec['node_id'])
        if spec.get('doa') is not None:
            self._die(spec['doa'], notify_control=False)

    def _context(self):
        ctx = api._ExecContext(**{k: getattr(self.ctx, k)
                                  for k in api._ExecContext.__slots__})
        ctx.task_id = api._new_task_id()
        api._set_context(ctx)

    def _actor_repr(self):
        try:
            return repr(self.instance)
        except Exception:
            return None

    @coro
    def construct(self, cls_desc, args, kwargs):
        if self.state != 'WAIT_CTOR':
            return
        self._context()
        name = self.spec['class_name'] + '.__init__'
        try:
            env_vars = (self.spec['runtime_env'] or {}).get('env_vars') or {}
            # NOTE: process-wide; actors that share a process share os.environ
            os.environ.update({k: str(v) for k, v in env_vars.items()})
            cls = resolve(cls_desc)
            args, kwargs = api._resolve_args(args, kwargs)
            instance = cls(*args, **kwargs)
        except api._DependencyFailed as e:
            err = api._dependency_error(name, e.error)
            self._die(self._creation_error(err), notify_control=True)
            return
        except Exception as e:
            err = _task_error(name, e, actor_id=self.spec['actor_id'])
            self._die(self._creation_error(err), notify_control=True)
            return
        if self.state == 'DEAD':   # killed while the constructor was blocked
            return
        self.instance = instance
        self.state = 'ALIVE'
        self._drain()

    def _creation_error(self, task_error):
        return ActorDiedError(
            cause=task_error.cause, actor_id=self.spec['actor_id'], actor_init_failed=True,
            error_msg='The actor died because of an error raised in its creation task, ' +
                      str(task_error))

    @coro
    def invoke(self, caller, seq, method_name, args, kwargs, return_refs):
        call = (method_name, args, kwargs, return_refs)
        expected = self.expected.get(caller, 0)
        if seq != expected:
            self.held.setdefault(caller, {})[seq] = call
            return
        self._enqueue(call)
        expected += 1
        held = self.held.get(caller)
        while held and expected in held:
            self._enqueue(held.pop(expected))
            expected += 1
        if held is not None and not held:
            del self.held[caller]
        self.expected[caller] = expected
        self._drain()

    def _enqueue(self, call):
        if self.state == 'DEAD':
            self._fail(call)
        else:
            self.queue.append(call)

    def _fail(self, call):
        for ref in call[3]:
            api._store_error(ref, self.death)

    def _drain(self):
        if self.running or self.state != 'ALIVE':
            return
        self.running = True
        try:
            while self.queue and self.state == 'ALIVE':
                self._execute(self.queue.popleft())
        finally:
            self.running = False
        if self.state == 'DEAD':
            while self.queue:
                self._fail(self.queue.popleft())

    def _execute(self, call):
        method_name, args, kwargs, refs = call
        self._context()
        name = self.spec['class_name'] + '.' + method_name
        try:
            args, kwargs = api._resolve_args(args, kwargs)
            if method_name == '__ray_ready__':
                result = True
            elif method_name == '__ray_terminate__':
                raise _ActorExit(0)
            else:
                result = getattr(self.instance, method_name)(*args, **kwargs)
        except api._DependencyFailed as e:
            err = api._dependency_error(name, e.error)
            for ref in refs:
                api._store_error(ref, err)
            return
        except SystemExit:   # exit_actor(), __ray_terminate__ or sys.exit()
            err = ActorDiedError(actor_id=self.spec['actor_id'], error_msg=(
                'The actor died unexpectedly before finishing this task.\n\tclass_name: ' +
                self.spec['class_name'] + '\n\tactor_id: ' + self.spec['actor_id'] +
                '\nThe actor is dead because its worker process has died. Worker exit '
                'type: INTENDED_USER_EXIT Worker exit detail: Worker exits by an user '
                'request. exit_actor() is called.'))
            if method_name == '__ray_terminate__':
                # graceful termination succeeds; later calls fail
                for ref in refs:
                    api._store(ref, None)
                refs = []
            self._die(err, notify_control=True, current_refs=refs)
            return
        except Exception as e:
            err = _task_error(name, e, actor_repr=self._actor_repr(),
                              actor_id=self.spec['actor_id'])
            for ref in refs:
                api._store_error(ref, err)
            return
        if self.state == 'DEAD':   # killed while this method was running
            for ref in refs:
                api._store_error(ref, self.death)
            return
        api._store_results(refs, result, name)

    def kill(self):
        if self.state != 'DEAD':
            self._die(ActorDiedError(actor_id=self.spec['actor_id'], error_msg=killed_message(
                self.spec['class_name'], self.spec['actor_id'])), notify_control=False)

    def _die(self, error, notify_control, current_refs=()):
        self.state = 'DEAD'
        self.death = error
        for ref in current_refs:
            api._store_error(ref, error)
        while self.queue:
            self._fail(self.queue.popleft())
        for calls in self.held.values():
            for call in calls.values():
                self._fail(call)
        self.held.clear()
        self.instance = None
        if notify_control:
            api._control.actor_died(self.spec['key'])


def _fits(avail, request):
    return all(avail.get(k, 0.0) + 1e-9 >= v for k, v in request.items())


def _take(avail, ids_pool, request):
    for k, v in request.items():
        avail[k] = avail.get(k, 0.0) - v
    ids = {}
    for k in ACCELERATOR_KEYS:
        n = int(request.get(k, 0))
        if n:
            ids[k] = ids_pool[k][:n]
            del ids_pool[k][:n]
    return ids


def _give(avail, ids_pool, request, ids):
    for k, v in request.items():
        avail[k] = avail.get(k, 0.0) + v
    for k, taken in ids.items():
        ids_pool.setdefault(k, []).extend(taken)
        ids_pool[k].sort()


class _RayControl(Chare):
    """The authority for names, resources and placement groups (on PE 0)."""

    def __init__(self):
        self.setMigratable(False)
        self.nodes = []
        self.actors_proxy = None
        self._reset()

    def _reset(self):
        self.actors = {}       # actor key -> record
        self.names = {}        # (namespace, name) -> actor key
        self.pgs = {}          # placement group id (hex) -> record
        self.pending = []      # ('actor', key) / ('pg', id) waiting for resources
        self.pe_load = collections.Counter()
        self.killed_early = set()

    def configure(self, nodes, actors_proxy):
        self.actors_proxy = actors_proxy
        self.nodes = []
        for i, node in enumerate(nodes):
            total = dict(node['Resources'])
            self.nodes.append({
                'index': i, 'node_id': node['NodeID'], 'pes': list(node['_pes']),
                'avail': dict(total),
                'ids': {k: list(range(int(total[k]))) for k in ACCELERATOR_KEYS
                        if total.get(k)}})
        self._reset()
        return True

    def shutdown(self):
        for key, rec in list(self.actors.items()):
            if rec['state'] != 'DEAD':
                self._kill(rec)
        for pg in self.pgs.values():
            pg['state'] = 'REMOVED'
        self._reset()
        return True

    # ---- queries

    def available_resources_per_node(self):
        return {n['node_id']: {k: v for k, v in n['avail'].items() if v > 1e-9}
                for n in self.nodes}

    def get_actor(self, name, namespace):
        key = self.names.get((namespace, name))
        if key is None:
            return None
        rec = self.actors[key]
        return rec['meta']['handle_state'] if rec['state'] != 'DEAD' else None

    # ---- actors

    def _driver_first_nodes(self):
        driver = api._charm().getPeHost(0)
        return sorted(self.nodes, key=lambda n: (n['index'] != driver, n['index']))

    def _choose_pe(self, node):
        pes = list(node['pes'])
        if len(pes) > 1 and 0 in pes:
            pes.remove(0)
        return min(pes, key=lambda pe: (self.pe_load[pe], pe))

    def create_actor(self, meta):
        key = meta['key']
        name = meta['name']
        if name is not None:
            name_key = (meta['namespace'], name)
            existing = self.names.get(name_key)
            if existing is not None and self.actors[existing]['state'] != 'DEAD':
                if meta['get_if_exists']:
                    return ('exists', self.actors[existing]['meta']['handle_state'])
                return ('taken',)
            self.names[name_key] = key
        rec = {'meta': meta, 'state': 'PENDING', 'node': None, 'pe': None, 'ids': {},
               'bundle': None}
        self.actors[key] = rec
        if key in self.killed_early:
            self.killed_early.discard(key)
            self._insert_dead(rec, killed_message(meta['class_name'], meta['actor_id']))
        elif not self._schedule_actor(rec):
            self.pending.append(('actor', key))
        return ('ok',)

    def _host_spec(self, rec, node_index, pg_id=None, bundle_specs=None, doa=None):
        meta = rec['meta']
        node_id = self.nodes[node_index]['node_id'] if node_index is not None else None
        return {'key': meta['key'], 'actor_id': meta['actor_id'],
                'class_name': meta['class_name'], 'name': meta['name'],
                'runtime_env': meta['runtime_env'],
                'accelerator_ids': {k: [str(i) for i in v] for k, v in rec['ids'].items()},
                'assigned_resources': dict(meta['request']), 'node_id': node_id,
                'pg_id': pg_id, 'pg_bundle_specs': bundle_specs, 'doa': doa}

    def _insert_dead(self, rec, message, pe=0):
        """Creates the actor's element only to fail the calls already sent to it."""
        meta = rec['meta']
        rec['state'] = 'DEAD'
        err = ActorDiedError(actor_id=meta['actor_id'], error_msg=message)
        self.actors_proxy.ckInsert(meta['key'], args=[self._host_spec(rec, None, doa=err)],
                                   onPE=pe)
        self._forget_name(rec)

    def _schedule_actor(self, rec):
        from .util.scheduling_strategies import (PlacementGroupSchedulingStrategy,
                                                 NodeAffinitySchedulingStrategy)
        meta = rec['meta']
        request = meta['request']
        strategy = meta['strategy']
        pg_id = bundle_specs = None
        if isinstance(strategy, PlacementGroupSchedulingStrategy):
            pg_id = strategy.placement_group.id.hex()
            pg = self.pgs.get(pg_id)
            if pg is None or pg['state'] == 'REMOVED':
                self._insert_dead(rec, 'The actor could not be scheduled: its placement '
                                  'group ' + pg_id + ' was removed or does not exist.')
                return True
            if pg['state'] != 'CREATED':
                return False
            index = strategy.placement_group_bundle_index
            indices = [index] if index >= 0 else range(len(pg['bundles']))
            for i in indices:
                bundle = pg['bundles'][i]
                if _fits(bundle['avail'], request):
                    rec['ids'] = _take(bundle['avail'], bundle['ids'], request)
                    rec['bundle'] = (pg_id, i)
                    node = self.nodes[bundle['node']]
                    bundle_specs = pg['specs']
                    break
            else:
                return False
        else:
            candidates = self._driver_first_nodes()
            if isinstance(strategy, NodeAffinitySchedulingStrategy):
                pinned = [n for n in self.nodes if n['node_id'] == strategy.node_id]
                candidates = pinned + (candidates if strategy.soft else [])
            for node in candidates:
                if _fits(node['avail'], request):
                    rec['ids'] = _take(node['avail'], node['ids'], request)
                    break
            else:
                return False
        pe = self._choose_pe(node)
        rec.update(state='ALIVE', node=node['index'], pe=pe)
        self.pe_load[pe] += 1
        self.actors_proxy.ckInsert(meta['key'], args=[self._host_spec(
            rec, node['index'], pg_id, bundle_specs)], onPE=pe)
        return True

    def _release(self, rec):
        request = rec['meta']['request']
        if rec['bundle'] is not None:
            pg = self.pgs.get(rec['bundle'][0])
            if pg is not None and pg['state'] == 'CREATED':
                bundle = pg['bundles'][rec['bundle'][1]]
                _give(bundle['avail'], bundle['ids'], request, rec['ids'])
        elif rec['node'] is not None:
            node = self.nodes[rec['node']]
            _give(node['avail'], node['ids'], request, rec['ids'])
        if rec['pe'] is not None:
            self.pe_load[rec['pe']] -= 1
        rec['ids'] = {}
        rec['bundle'] = None

    def _forget_name(self, rec):
        name = rec['meta']['name']
        if name is not None:
            name_key = (rec['meta']['namespace'], name)
            if self.names.get(name_key) == rec['meta']['key']:
                del self.names[name_key]

    def _kill(self, rec):
        if rec['state'] == 'PENDING':
            self.pending = [p for p in self.pending if p != ('actor', rec['meta']['key'])]
            self._insert_dead(rec, killed_message(rec['meta']['class_name'],
                                                  rec['meta']['actor_id']))
            return
        if rec['state'] == 'ALIVE':
            self.actors_proxy[rec['meta']['key']].kill()
            self._release(rec)
        rec['state'] = 'DEAD'
        self._forget_name(rec)

    def kill_actor(self, key):
        rec = self.actors.get(key)
        if rec is None:
            # the creation request has not arrived yet
            self.killed_early.add(key)
            return True
        self._kill(rec)
        self._schedule_pending()
        return True

    def actor_died(self, key):
        rec = self.actors.get(key)
        if rec is None or rec['state'] == 'DEAD':
            return
        self._release(rec)
        rec['state'] = 'DEAD'
        self._forget_name(rec)
        self._schedule_pending()

    def _schedule_pending(self):
        progress = True
        while progress and self.pending:
            progress = False
            for item in list(self.pending):
                kind, key = item
                if kind == 'actor':
                    done = self._schedule_actor(self.actors[key])
                else:
                    done = self._schedule_pg(self.pgs[key])
                if done:
                    self.pending.remove(item)
                    progress = True

    # ---- placement groups

    def create_pg(self, pg_id, specs, strategy, name):
        pg = {'id': pg_id, 'specs': specs, 'strategy': strategy, 'name': name,
              'state': 'PENDING', 'bundles': None, 'ready_refs': []}
        self.pgs[pg_id] = pg
        if not self._schedule_pg(pg):
            self.pending.append(('pg', pg_id))

    def _assign_bundles(self, specs, strategy):
        """Node index for each bundle, or None if they cannot all be placed now.
        Works on copies so a partial assignment never reserves anything."""
        avail = {n['index']: dict(n['avail']) for n in self.nodes}
        order = [n['index'] for n in self._driver_first_nodes()]

        def place(i, node):
            if _fits(avail[node], specs[i]):
                for k, v in specs[i].items():
                    avail[node][k] = avail[node].get(k, 0.0) - v
                return True
            return False

        if strategy in ('PACK', 'STRICT_PACK'):
            for node in order:
                saved = {k: dict(v) for k, v in avail.items()}
                if all(place(i, node) for i in range(len(specs))):
                    return [node] * len(specs)
                avail = saved
            if strategy == 'STRICT_PACK':
                return None
            assignment = []
            for i in range(len(specs)):
                preferred = ([assignment[-1]] if assignment else []) + order
                for node in preferred:
                    if place(i, node):
                        assignment.append(node)
                        break
                else:
                    return None
            return assignment
        if strategy == 'SPREAD':
            assignment = []
            for i in range(len(specs)):
                ranked = sorted(order, key=lambda n: (assignment.count(n), order.index(n)))
                for node in ranked:
                    if place(i, node):
                        assignment.append(node)
                        break
                else:
                    return None
            return assignment
        # STRICT_SPREAD
        assignment = []
        for i in range(len(specs)):
            for node in order:
                if node not in assignment and place(i, node):
                    assignment.append(node)
                    break
            else:
                return None
        return assignment

    def _schedule_pg(self, pg):
        if pg['state'] != 'PENDING':
            return True
        assignment = self._assign_bundles(pg['specs'], pg['strategy'])
        if assignment is None:
            return False
        bundles = []
        for spec, node_index in zip(pg['specs'], assignment):
            node = self.nodes[node_index]
            ids = _take(node['avail'], node['ids'], spec)
            bundles.append({'node': node_index, 'avail': dict(spec), 'ids': ids,
                            'all_ids': {k: list(v) for k, v in ids.items()}})
        pg['bundles'] = bundles
        pg['state'] = 'CREATED'
        for ref in pg['ready_refs']:
            api._store(ref, self._pg_object(pg))
        pg['ready_refs'] = []
        return True

    def _pg_object(self, pg):
        from .util.placement_group import PlacementGroup, PlacementGroupID
        return PlacementGroup(PlacementGroupID(pg['id']), pg['specs'])

    def pg_ready(self, pg_id, ref):
        pg = self.pgs.get(pg_id)
        if pg is None or pg['state'] == 'REMOVED':
            api._store_error(ref, RaySystemError('Placement group ' + pg_id +
                                                 ' was removed'))
        elif pg['state'] == 'CREATED':
            api._store(ref, self._pg_object(pg))
        else:
            pg['ready_refs'].append(ref)

    def remove_pg(self, pg_id):
        pg = self.pgs.get(pg_id)
        if pg is None or pg['state'] == 'REMOVED':
            return True
        if pg['state'] == 'PENDING':
            self.pending = [p for p in self.pending if p != ('pg', pg_id)]
        else:
            for rec in self.actors.values():
                if rec['state'] != 'DEAD' and rec['bundle'] is not None and \
                        rec['bundle'][0] == pg_id:
                    self._kill(rec)
            for spec, bundle in zip(pg['specs'], pg['bundles']):
                node = self.nodes[bundle['node']]
                _give(node['avail'], node['ids'], spec, bundle['all_ids'])
        for ref in pg['ready_refs']:
            api._store_error(ref, RaySystemError('Placement group ' + pg_id + ' was removed'))
        pg['ready_refs'] = []
        pg['state'] = 'REMOVED'
        self._schedule_pending()
        return True

    def placement_group_bundle_nodes(self, pg_id):
        return [b['node'] for b in self.pgs[pg_id]['bundles']]

    def _pg_table_entry(self, pg):
        to_node = {}
        if pg['bundles'] is not None and pg['state'] == 'CREATED':
            to_node = {i: self.nodes[b['node']]['node_id'] for i, b in enumerate(pg['bundles'])}
        return {'placement_group_id': pg['id'], 'name': pg['name'],
                'bundles': {i: dict(s) for i, s in enumerate(pg['specs'])},
                'bundles_to_node_id': to_node, 'strategy': pg['strategy'],
                'state': pg['state'],
                'stats': {'scheduling_state': 'FINISHED' if pg['state'] == 'CREATED'
                          else 'PENDING'}}

    def pg_table(self, pg_id):
        if pg_id is None:
            return {i: self._pg_table_entry(pg) for i, pg in self.pgs.items()}
        pg = self.pgs.get(pg_id)
        return self._pg_table_entry(pg) if pg is not None else {}

    def pg_by_name(self, name):
        for pg in self.pgs.values():
            if pg['name'] == name and pg['state'] != 'REMOVED':
                return (pg['id'], pg['specs'])
        return None
