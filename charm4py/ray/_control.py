"""The control plane: actor placement, node resources, names and placement
groups, spread over the PEs instead of kept by one central chare (Ray keeps
all of it in its GCS and asks raylets to grant leases).

``_Control`` is a Group: one element per PE, which plays up to three roles.

- **Coordinator** of the actors and placement groups created on its PE (by the
  driver, a task or an actor that PE serves). It owns their life cycle -- state,
  kill, death, waiting for resources -- so there is no global actor table. A
  handle names its actor's coordinator: the first element of the actor's key.
- **Node controller**, on the last PE of each node (physical host; the last,
  to keep it off PE 0, which serves the driver): the only
  authority for that node's resources and accelerator IDs. It places actors on
  its PEs (and creates their elements there), and holds the bundles of
  placement groups reserved on the node.
- **Name shard**: actor and placement group names are spread over all PEs by
  hash.

Placing an actor asks node controllers in turn (its own node first, as Ray's
default policy prefers the local node); one that has room reserves it at once,
so a placement never needs a global view. Creating a placement group reserves
its bundles in two phases, involving only the nodes they go to: reserve on each
(each node atomically), then commit; if any node refuses, the others are
released. Work that does not fit yet waits at its coordinator, and node
controllers tell the coordinators waiting on them when they free resources."""

import collections
import uuid

from charm4py import Chare, coro
from . import api
from ._options import ACCELERATOR_KEYS
from .actor import killed_message
from .exceptions import ActorDiedError, RaySystemError


def _fits(avail, request):
    return all(avail.get(k, 0.0) + 1e-9 >= v for k, v in request.items())


def _id_order(device_id):
    return (0, int(device_id), '') if device_id.isdigit() else (1, 0, device_id)


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
        ids_pool[k].sort(key=_id_order)


def name_shard(key):
    """The PE whose element keeps the name ``key`` (a tuple)."""
    import zlib
    return zlib.crc32(repr(key).encode()) % api._charm().numPes()


def pg_coordinator(pg_id):
    """A placement group's ID starts with its coordinator's PE."""
    return int(pg_id[:8], 16)


def new_pg_id(job_id):
    # 18 bytes, like Ray's: the coordinator PE, 10 random bytes, the job
    return format(api._charm().myPe(), '08x') + uuid.uuid4().hex[:20] + job_id


def node_pe(node_index):
    """The PE of a node's controller."""
    return api._nodes[node_index]['_pes'][-1]


def local():
    """This PE's element."""
    return api._ctl[api._charm().myPe()].ckLocalBranch()


class _Control(Chare):

    def __init__(self):
        self.actors_proxy = None
        self.node = None          # this PE is its node's controller
        self._reset()

    def _reset(self):
        # coordinator
        self.actors = {}          # actor key -> record
        self.pgs = {}             # pg id -> record
        self.waiting = []         # ('actor', key) / ('pg', id), in arrival order
        self.retry_scheduled = False
        # counts resources_freed notifications: an attempt that fails retries at
        # once if one arrived while it was in flight (it found nothing waiting
        # to retry, and the nodes have already forgotten this coordinator)
        self.freed_epoch = 0
        self.spread_next = 0      # node offset of this coordinator's next SPREAD actor
        # name shard
        self.names = {}           # name key -> (owner key or pg id, value)
        # node controller
        if self.node is not None:
            self.node = self._fresh_node(self.node['config'])

    # ------------------------------------------------------------- setup

    def configure(self, nodes, actors_proxy):
        """ray.init (broadcast): the node table and the actor array."""
        self.actors_proxy = actors_proxy
        me = api._charm().myPe()
        self.node = None
        for index, node in enumerate(nodes):
            if node['_pes'][-1] == me:
                self.node = self._fresh_node({'index': index, 'node_id': node['NodeID'],
                                              'pes': list(node['_pes']),
                                              'total': dict(node['Resources']),
                                              'gpu_ids': list(node['_gpu_ids'])})
        self._reset()

    def _fresh_node(self, config):
        total = config['total']
        return {'config': config, 'index': config['index'], 'avail': dict(total),
                'ids': {k: (list(config['gpu_ids']) if k == 'GPU' else
                            [str(i) for i in range(int(total[k]))])
                        for k in ACCELERATOR_KEYS if total.get(k)},
                'pe_load': collections.Counter(),
                'actors': {},     # actor key -> (pe, request, ids, bundle key or None)
                'bundles': {},    # (pg id, index) -> {'spec', 'avail', 'ids', 'all_ids'}
                'removed_pgs': set(),
                'watchers': set()}

    def shutdown(self):
        """The runtime stops (broadcast): kill what is left, forget everything."""
        for rec in list(self.actors.values()):
            if rec['state'] == 'ALIVE':
                self.actors_proxy[rec['meta']['key']].kill()
        self._reset()

    # ------------------------------------------------------------- names

    def claim_name(self, name_key, owner, value, get_if_exists):
        """Takes a name. Returns ('ok',), ('exists', value) or ('taken',)."""
        held = self.names.get(name_key)
        if held is not None:
            return ('exists', held[1]) if get_if_exists else ('taken',)
        self.names[name_key] = (owner, value)
        return ('ok',)

    def lookup_name(self, name_key):
        held = self.names.get(name_key)
        return held[1] if held is not None else None

    def release_name(self, name_key, owner):
        held = self.names.get(name_key)
        if held is not None and held[0] == owner:
            del self.names[name_key]

    def _claim(self, name_key, owner, value, get_if_exists=False):
        return self.thisProxy[name_shard(name_key)].claim_name(
            name_key, owner, value, get_if_exists, ret=True).get()

    def _release_name(self, name_key, owner):
        self.thisProxy[name_shard(name_key)].release_name(name_key, owner)

    # ------------------------------------------------------------- node role

    def _choose_pe(self, node):
        pes = list(node['config']['pes'])
        if len(pes) > 1 and 0 in pes:
            pes.remove(0)    # PE 0 serves the driver
        return min(pes, key=lambda pe: (node['pe_load'][pe], pe))

    def place_actor(self, key, request, bundle_key, spec, coordinator):
        """Reserves resources for an actor on this node and creates its element
        on one of the node's PEs. Returns (pe, accelerator ids), 'removed' if its
        placement group is gone, or None if it does not fit now (the coordinator
        is told when resources are freed)."""
        node = self.node
        if bundle_key is not None:
            bundle = node['bundles'].get(bundle_key)
            if bundle is None:
                if bundle_key[0] in node['removed_pgs']:
                    return 'removed'
                node['watchers'].add(coordinator)
                return None
            if not _fits(bundle['avail'], request):
                node['watchers'].add(coordinator)
                return None
            ids = _take(bundle['avail'], bundle['ids'], request)
        else:
            if not _fits(node['avail'], request):
                node['watchers'].add(coordinator)
                return None
            ids = _take(node['avail'], node['ids'], request)
        pe = self._choose_pe(node)
        node['pe_load'][pe] += 1
        node['actors'][key] = (pe, dict(request), ids, bundle_key)
        spec = dict(spec, node_id=node['config']['node_id'],
                    accelerator_ids={k: [str(i) for i in v] for k, v in ids.items()})
        self.actors_proxy.ckInsert(key, args=[spec], onPE=pe)
        return (pe, ids)

    def release_actor(self, key):
        """An actor placed on this node ended: return its resources."""
        node = self.node
        placed = node['actors'].pop(key, None)
        if placed is None:
            return
        pe, request, ids, bundle_key = placed
        node['pe_load'][pe] -= 1
        if bundle_key is not None:
            bundle = node['bundles'].get(bundle_key)
            if bundle is not None:
                _give(bundle['avail'], bundle['ids'], request, ids)
        else:
            _give(node['avail'], node['ids'], request, ids)
        self._notify_watchers()

    def _notify_watchers(self):
        watchers, self.node['watchers'] = self.node['watchers'], set()
        for pe in watchers:
            self.thisProxy[pe].resources_freed()

    def node_available(self, watcher=None):
        """(node id, available resources). ``watcher``: a coordinator to tell when
        resources are freed after this snapshot."""
        node = self.node
        if watcher is not None:
            node['watchers'].add(watcher)
        return (node['config']['node_id'],
                {k: v for k, v in node['avail'].items() if v > 1e-9})

    def reserve_bundles(self, pg_id, items, coordinator):
        """Phase one of placing a placement group: reserve these bundles
        [(index, spec)] on this node, all or none. Returns True or False."""
        node = self.node
        avail = dict(node['avail'])
        for _, spec in items:
            if not _fits(avail, spec):
                node['watchers'].add(coordinator)
                return False
            for k, v in spec.items():
                avail[k] = avail.get(k, 0.0) - v
        for index, spec in items:
            ids = _take(node['avail'], node['ids'], spec)
            node['bundles'][(pg_id, index)] = {
                'spec': dict(spec), 'avail': dict(spec), 'ids': ids,
                'all_ids': {k: list(v) for k, v in ids.items()}, 'committed': False}
        return True

    def commit_bundles(self, pg_id):
        """Phase two: the whole placement group is reserved."""
        for (bundle_pg, _), bundle in self.node['bundles'].items():
            if bundle_pg == pg_id:
                bundle['committed'] = True

    @coro
    def release_bundles(self, pg_id, removed):
        """Gives back this node's bundles of a placement group: an aborted
        reservation, or (``removed``) a removed group, whose actors die. Returns
        once they are dead."""
        node = self.node
        if removed:
            node['removed_pgs'].add(pg_id)
            killed = []
            for key, placed in list(node['actors'].items()):
                if placed[3] is not None and placed[3][0] == pg_id:
                    killed.append(self.actors_proxy[key].kill(ret=True))
                    self.release_actor(key)
                    self.thisProxy[key[0]].actor_died(key)
            for done in killed:
                done.get()
        for bundle_key in [b for b in node['bundles'] if b[0] == pg_id]:
            bundle = node['bundles'].pop(bundle_key)
            _give(node['avail'], node['ids'], bundle['spec'], bundle['all_ids'])
        self._notify_watchers()
        return True

    # ------------------------------------------------------------- actors

    def register_actor(self, meta):
        """Called directly by the creator, on this PE, before it hands out the
        handle (so a kill can never come first). Its name, if any, is claimed."""
        self.actors[meta['key']] = {'meta': meta, 'state': 'PENDING', 'node': None,
                                    'pe': None, 'placing': False}
        self.thisProxy[api._charm().myPe()].place(meta['key'])

    @coro
    def place(self, key):
        """Places a registered actor, waiting for resources if needed."""
        rec = self.actors.get(key)
        if rec is not None:
            self._try_place(rec)

    def _insert_dead(self, rec, error):
        """Creates the actor's element only to fail the calls sent to it."""
        meta = rec['meta']
        rec['state'] = 'DEAD'
        self.actors_proxy.ckInsert(meta['key'], args=[self._spec(meta, doa=error)],
                                   onPE=api._charm().myPe())
        self._forget(rec)

    def _spec(self, meta, pg_id=None, bundle_specs=None, doa=None):
        return {'key': meta['key'], 'actor_id': meta['actor_id'],
                'class_name': meta['class_name'], 'name': meta['name'],
                'runtime_env': meta['runtime_env'], 'job': meta['job'],
                'accelerator_ids': {}, 'assigned_resources': dict(meta['request']),
                'node_id': None, 'pg_id': pg_id, 'pg_bundle_specs': bundle_specs,
                'doa': doa}

    def _forget(self, rec):
        meta = rec['meta']
        if meta['name'] is not None:
            self._release_name(('actor', meta['namespace'], meta['name']), meta['key'])
        self.actors.pop(meta['key'], None)

    def _candidates(self, rec):
        """[(node index, bundle key)] to try, in order; or an error message."""
        from .util.scheduling_strategies import (PlacementGroupSchedulingStrategy,
                                                 NodeAffinitySchedulingStrategy)
        strategy = rec['meta']['strategy']
        if isinstance(strategy, PlacementGroupSchedulingStrategy):
            pg_id = strategy.placement_group.id.hex()
            state = self.thisProxy[pg_coordinator(pg_id)].pg_state(pg_id, ret=True).get()
            if state is None or state[0] == 'REMOVED':
                return ('The actor could not be scheduled: its placement group ' + pg_id +
                        ' was removed or does not exist.')
            if state[0] == 'PENDING':
                # wait for it; the group's coordinator tells us when it is created
                self.thisProxy[pg_coordinator(pg_id)].watch_pg(pg_id, api._charm().myPe())
                return []
            nodes, specs = state[1], state[2]
            index = strategy.placement_group_bundle_index
            rec['pg'] = (pg_id, specs)
            indices = [index] if index >= 0 else range(len(nodes))
            return [(nodes[i], (pg_id, i)) for i in indices]
        here = api._my_host()
        if strategy == 'SPREAD':
            # Ray spreads these over the nodes, best effort. Each coordinator
            # deals its SPREAD actors round robin, starting with its own node;
            # an actor keeps its starting node when it has to try again.
            if 'spread_from' not in rec:
                rec['spread_from'] = (here + self.spread_next) % len(api._nodes)
                self.spread_next += 1
            here = rec['spread_from']
        order = sorted(range(len(api._nodes)),
                       key=lambda n: ((n - here) % len(api._nodes)))
        if isinstance(strategy, NodeAffinitySchedulingStrategy):
            pinned = [i for i, n in enumerate(api._nodes)
                      if n['NodeID'] == strategy.node_id]
            order = pinned + ([n for n in order if n not in pinned] if strategy.soft else [])
        return [(n, None) for n in order]

    def _try_place(self, rec):
        """Tries the candidate nodes in turn (this runs as a coroutine)."""
        if rec['state'] != 'PENDING' or rec['placing']:
            return
        rec['placing'] = True
        try:
            while self._place_once(rec):
                pass
        finally:
            rec['placing'] = False

    def _place_once(self, rec):
        """One placement attempt. Returns True to try again at once."""
        epoch = self.freed_epoch
        candidates = self._candidates(rec)
        if isinstance(candidates, str):
            self._insert_dead(rec, ActorDiedError(actor_id=rec['meta']['actor_id'],
                                                  error_msg=candidates))
            return
        meta = rec['meta']
        pg_id, specs = rec.get('pg', (None, None))
        spec = self._spec(meta, pg_id, specs)
        me = api._charm().myPe()
        for node_index, bundle_key in candidates:
            placed = self.thisProxy[node_pe(node_index)].place_actor(
                meta['key'], meta['request'], bundle_key, spec, me, ret=True).get()
            if placed == 'removed':
                self._insert_dead(rec, ActorDiedError(
                    actor_id=meta['actor_id'], error_msg='The actor could not be '
                    'scheduled: its placement group was removed.'))
                return
            if placed is not None:
                if rec['state'] == 'KILLED':
                    # killed while it was being placed
                    self.actors_proxy[meta['key']].kill()
                    self.thisProxy[node_pe(node_index)].release_actor(meta['key'])
                    self._forget(rec)
                    return
                rec.update(state='ALIVE', node=node_index, pe=placed[0])
                return
        if rec['state'] == 'KILLED':
            self._insert_dead(rec, ActorDiedError(
                actor_id=meta['actor_id'],
                error_msg=killed_message(meta['class_name'], meta['actor_id'])))
            return
        if self.freed_epoch != epoch:
            return True
        self.waiting.append(('actor', meta['key']))

    def kill_actor(self, key):
        rec = self.actors.get(key)
        if rec is None:
            return True
        if rec['state'] == 'ALIVE':
            self.actors_proxy[key].kill()
            self.thisProxy[node_pe(rec['node'])].release_actor(key)
            rec['state'] = 'DEAD'
            self._forget(rec)
        elif rec['state'] == 'PENDING':
            if rec['placing']:
                rec['state'] = 'KILLED'   # _try_place finishes the job
            else:
                self._drop_waiting(('actor', key))
                self._insert_dead(rec, ActorDiedError(
                    actor_id=rec['meta']['actor_id'],
                    error_msg=killed_message(rec['meta']['class_name'],
                                             rec['meta']['actor_id'])))
        return True

    def actor_died(self, key):
        """The actor ended (crash, exit_actor, its group was removed)."""
        rec = self.actors.get(key)
        if rec is None or rec['state'] == 'DEAD':
            return
        if rec['state'] == 'ALIVE' and rec['node'] is not None:
            self.thisProxy[node_pe(rec['node'])].release_actor(key)
        rec['state'] = 'DEAD'
        self._forget(rec)

    def _drop_waiting(self, item):
        self.waiting = [w for w in self.waiting if w != item]

    def resources_freed(self):
        """A node this coordinator waits on freed resources: retry, once per
        burst of notifications."""
        self.freed_epoch += 1
        if not self.retry_scheduled:
            self.retry_scheduled = True
            self.thisProxy[api._charm().myPe()].retry_waiting()

    @coro
    def retry_waiting(self):
        self.retry_scheduled = False
        waiting, self.waiting = self.waiting, []
        for kind, key in waiting:
            if kind == 'actor':
                rec = self.actors.get(key)
                if rec is not None:
                    self._try_place(rec)
            else:
                pg = self.pgs.get(key)
                if pg is not None:
                    self._try_reserve(pg)

    @coro
    def end_job(self, job_id):
        """The job ended (broadcast): its actors and placement groups go."""
        for key, rec in list(self.actors.items()):
            if rec['meta']['job']['job_id'] == job_id:
                self.kill_actor(key)
        for pg_id, pg in list(self.pgs.items()):
            if pg['job_id'] == job_id:
                self._remove_pg(pg)
        return True

    # ------------------------------------------------------------- placement groups

    @coro
    def create_pg(self, pg_id, specs, strategy, name, job_id):
        pg = {'id': pg_id, 'specs': specs, 'strategy': strategy, 'name': name,
              'state': 'PENDING', 'nodes': None, 'ready_refs': [], 'job_id': job_id,
              'watchers': set(), 'reserving': False}
        self.pgs[pg_id] = pg
        self._try_reserve(pg)

    def _view(self):
        """Every node's available resources, now (possibly stale by the time
        they are used: each node checks its reservation again). Each node will
        tell this coordinator when it frees resources after its snapshot."""
        me = api._charm().myPe()
        replies = [self.thisProxy[node_pe(i)].node_available(me, ret=True)
                   for i in range(len(api._nodes))]
        return [dict(reply.get()[1]) for reply in replies]

    def _try_reserve(self, pg):
        if pg['state'] != 'PENDING' or pg['reserving']:
            return
        pg['reserving'] = True
        try:
            while self._reserve_once(pg):
                pass
        finally:
            pg['reserving'] = False

    def _reserve_once(self, pg):
        """One reservation attempt. Returns True to try again at once."""
        epoch = self.freed_epoch
        assignment = _assign_bundles(self._view(), pg['specs'], pg['strategy'])
        me = api._charm().myPe()
        if assignment is not None:
            per_node = collections.defaultdict(list)
            for index, node_index in enumerate(assignment):
                per_node[node_index].append((index, pg['specs'][index]))
            replies = {n: self.thisProxy[node_pe(n)].reserve_bundles(
                pg['id'], items, me, ret=True) for n, items in per_node.items()}
            reserved = [n for n, reply in replies.items() if reply.get()]
            if len(reserved) == len(per_node) and pg['state'] == 'PENDING':
                for n in reserved:
                    self.thisProxy[node_pe(n)].commit_bundles(pg['id'])
                pg['nodes'] = assignment
                pg['state'] = 'CREATED'
                self._pg_created(pg)
                return
            for n in reserved:   # abort
                self.thisProxy[node_pe(n)].release_bundles(pg['id'], False)
            if pg['state'] == 'PENDING':
                # a node refused: the view was stale (other groups reserved in
                # the meantime), not necessarily full. Look again rather than
                # wait for a node to free resources, which the nodes with room
                # may never do.
                return True
        if pg['state'] == 'PENDING':
            if self.freed_epoch != epoch:
                return True
            # wait: every node will say when it frees resources (see _view)
            self.waiting.append(('pg', pg['id']))

    def _pg_created(self, pg):
        from .util.placement_group import PlacementGroup, PlacementGroupID
        for ref in pg['ready_refs']:
            api._store(ref, PlacementGroup(PlacementGroupID(pg['id']), pg['specs']))
        pg['ready_refs'] = []
        watchers, pg['watchers'] = pg['watchers'], set()
        for pe in watchers:
            self.thisProxy[pe].resources_freed()

    def watch_pg(self, pg_id, coordinator):
        pg = self.pgs.get(pg_id)
        if pg is None or pg['state'] != 'PENDING':
            self.thisProxy[coordinator].resources_freed()
        else:
            pg['watchers'].add(coordinator)

    def pg_state(self, pg_id):
        """(state, bundle node indices, specs), or None if unknown here."""
        pg = self.pgs.get(pg_id)
        if pg is None:
            return None
        return (pg['state'], pg['nodes'], pg['specs'])

    def pg_ready(self, pg_id, ref):
        from .util.placement_group import PlacementGroup, PlacementGroupID
        pg = self.pgs.get(pg_id)
        if pg is None or pg['state'] == 'REMOVED':
            api._store_error(ref, RaySystemError('Placement group ' + pg_id + ' was removed'))
        elif pg['state'] == 'CREATED':
            api._store(ref, PlacementGroup(PlacementGroupID(pg['id']), pg['specs']))
        else:
            pg['ready_refs'].append(ref)

    @coro
    def remove_pg(self, pg_id):
        pg = self.pgs.get(pg_id)
        if pg is not None:
            self._remove_pg(pg)
        return True

    def _remove_pg(self, pg):
        """Removes a group; returns once its actors are dead (a coroutine)."""
        if pg['state'] == 'REMOVED':
            return
        if pg['state'] == 'CREATED':
            nodes = set(pg['nodes'])
        else:
            self._drop_waiting(('pg', pg['id']))
            # a reservation in progress may hold bundles anywhere
            nodes = range(len(api._nodes))
        pg['state'] = 'REMOVED'
        done = [self.thisProxy[node_pe(n)].release_bundles(pg['id'], True, ret=True)
                for n in nodes]
        for reply in done:
            reply.get()
        for ref in pg['ready_refs']:
            api._store_error(ref, RaySystemError('Placement group ' + pg['id'] +
                                                 ' was removed'))
        pg['ready_refs'] = []
        pg['state'] = 'REMOVED'
        if pg['name']:
            self._release_name(('pg', pg['name']), pg['id'])
        watchers, pg['watchers'] = pg['watchers'], set()
        for pe in watchers:
            self.thisProxy[pe].resources_freed()

    def pg_table(self, pg_id):
        """This coordinator's groups (``pg_id`` None) or one of them."""
        def entry(pg):
            to_node = {}
            if pg['state'] == 'CREATED':
                to_node = {i: api._nodes[n]['NodeID'] for i, n in enumerate(pg['nodes'])}
            return {'placement_group_id': pg['id'], 'name': pg['name'],
                    'bundles': {i: dict(s) for i, s in enumerate(pg['specs'])},
                    'bundles_to_node_id': to_node, 'strategy': pg['strategy'],
                    'state': pg['state'],
                    'stats': {'scheduling_state': 'FINISHED' if pg['state'] == 'CREATED'
                              else 'PENDING'}}
        if pg_id is None:
            return {i: entry(pg) for i, pg in self.pgs.items()}
        pg = self.pgs.get(pg_id)
        return entry(pg) if pg is not None else {}

    def pg_bundle_nodes(self, pg_id):
        return list(self.pgs[pg_id]['nodes'])


def _assign_bundles(avail_by_node, specs, strategy):
    """Node index for each bundle, given every node's available resources, or
    None if they cannot all be placed now. Nodes are tried from the driver's
    (PE 0's) on, as the centralized scheduler did."""
    avail = {i: dict(a) for i, a in enumerate(avail_by_node)}
    driver = api._pe_host(0)
    order = sorted(avail, key=lambda n: (n != driver, n))

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
    assignment = []   # STRICT_SPREAD
    for i in range(len(specs)):
        for node in order:
            if node not in assignment and place(i, node):
                assignment.append(node)
                break
        else:
            return None
    return assignment
