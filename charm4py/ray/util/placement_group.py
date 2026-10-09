"""Placement groups: atomic reservations of resource bundles on nodes."""

import numbers
import uuid

from .. import api
from .._options import _check_resource_amount

VALID_PLACEMENT_GROUP_STRATEGIES = {'PACK', 'SPREAD', 'STRICT_PACK', 'STRICT_SPREAD'}


class PlacementGroupID(object):

    def __init__(self, hex_id):
        self._hex = hex_id

    def hex(self):
        return self._hex

    def binary(self):
        return bytes.fromhex(self._hex)

    def is_nil(self):
        return False

    def __eq__(self, other):
        return isinstance(other, PlacementGroupID) and other._hex == self._hex

    def __hash__(self):
        return hash(self._hex)

    def __repr__(self):
        return 'PlacementGroupID(' + self._hex + ')'


class PlacementGroup(object):
    """A handle to a placement group."""

    def __init__(self, id, bundle_cache=None):
        self.id = id
        self.bundle_cache = bundle_cache

    @staticmethod
    def empty():
        return PlacementGroup(PlacementGroupID('ff' * 18))

    def ready(self):
        """An ObjectRef that becomes ready (holding this placement group) once every
        bundle has been reserved."""
        api._ensure_initialized()
        ref = api._new_ref()
        api._control.pg_ready(self.id.hex(), ref)
        return ref

    def wait(self, timeout_seconds=30):
        ready, _ = api.wait([self.ready()], timeout=timeout_seconds)
        return len(ready) == 1

    @property
    def bundle_specs(self):
        if self.bundle_cache is None:
            table = placement_group_table(self)
            self.bundle_cache = [table['bundles'][i] for i in sorted(table['bundles'])]
        return [dict(b) for b in self.bundle_cache]

    @property
    def bundle_count(self):
        return len(self.bundle_specs)

    def __eq__(self, other):
        return isinstance(other, PlacementGroup) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return '<PlacementGroup ' + repr(self.id) + '>'


def _validate_bundles(bundles):
    if not isinstance(bundles, list):
        raise ValueError('The type of bundles must be list, got ' + str(bundles))
    if not bundles:
        raise ValueError('Bundles must be a non-empty list of resource dictionaries. '
                         'For example: `[{"CPU": 1.0}, {"GPU": 1.0}]`. Got empty list '
                         'instead.')
    specs = []
    for bundle in bundles:
        if not isinstance(bundle, dict) or \
                not all(isinstance(v, numbers.Number) for v in bundle.values()):
            raise ValueError('Bundles must be a list of resource dictionaries, got ' +
                             str(bundles))
        if not any(v > 0 for v in bundle.values()):
            raise ValueError('Bundles cannot be an empty dictionary or resources with '
                             'only 0 values. Bundles: ' + str(bundles))
        for key, amount in bundle.items():
            _check_resource_amount(key, amount)
        specs.append({k: float(v) for k, v in bundle.items() if v > 0})
    return specs


def placement_group(bundles, strategy='PACK', name='', lifetime=None,
                    _soft_target_node_id=None, bundle_label_selector=None, **kwargs):
    """Asynchronously reserves ``bundles``; ``ray.get(pg.ready())`` waits for it."""
    specs = _validate_bundles(bundles)
    if strategy not in VALID_PLACEMENT_GROUP_STRATEGIES:
        raise ValueError('Invalid placement group strategy ' + str(strategy) +
                         '. Supported strategies are: ' +
                         str(VALID_PLACEMENT_GROUP_STRATEGIES) + '.')
    if lifetime not in (None, 'detached'):
        raise ValueError("Placement group `lifetime` argument must be either `None` or "
                         "'detached'. Got " + str(lifetime) + '.')
    if lifetime == 'detached' or _soft_target_node_id is not None or bundle_label_selector:
        raise NotImplementedError('charm4py.ray does not support detached placement '
                                  'groups, soft target nodes or bundle label selectors')
    for key in kwargs:
        if not key.startswith('_'):
            raise TypeError("placement_group() got an unexpected keyword argument '" +
                            key + "'")
    api._ensure_initialized()
    pg_id = uuid.uuid4().hex[:28] + api.JOB_ID
    api._control.create_pg(pg_id, specs, strategy, name)
    return PlacementGroup(PlacementGroupID(pg_id), specs)


def remove_placement_group(placement_group):
    """Releases the reservation and kills the actors scheduled in it."""
    api._ensure_initialized()
    api._control.remove_pg(placement_group.id.hex(), ret=True).get()


def get_placement_group(placement_group_name):
    api._ensure_initialized()
    found = api._control.pg_by_name(placement_group_name, ret=True).get()
    if found is None:
        raise ValueError('Failed to look up placement group with name: ' +
                         str(placement_group_name))
    return PlacementGroup(PlacementGroupID(found[0]), found[1])


def get_current_placement_group():
    """The placement group the current task or actor was scheduled in, or None
    (always None in the driver)."""
    ctx = api._current_context()
    if ctx is None or ctx.pg_id is None:
        return None
    return PlacementGroup(PlacementGroupID(ctx.pg_id), ctx.pg_bundle_specs)


def placement_group_table(placement_group=None):
    api._ensure_initialized()
    pg_id = placement_group.id.hex() if placement_group is not None else None
    return api._control.pg_table(pg_id, ret=True).get()
