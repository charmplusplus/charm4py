"""Scheduling strategies for tasks and actors."""


class PlacementGroupSchedulingStrategy(object):
    """Schedule in a bundle of a placement group (``placement_group_bundle_index``
    -1 means any bundle with room)."""

    def __init__(self, placement_group, placement_group_bundle_index=-1,
                 placement_group_capture_child_tasks=None):
        self.placement_group = placement_group
        self.placement_group_bundle_index = placement_group_bundle_index
        self.placement_group_capture_child_tasks = placement_group_capture_child_tasks


class NodeAffinitySchedulingStrategy(object):
    """Schedule on the node with ID ``node_id``; if ``soft``, elsewhere when that
    node does not have the resources."""

    def __init__(self, node_id, soft, _spill_on_unavailable=False,
                 _fail_on_unavailable=False):
        self.node_id = node_id
        self.soft = soft
        self._spill_on_unavailable = _spill_on_unavailable
        self._fail_on_unavailable = _fail_on_unavailable


class NodeLabelSchedulingStrategy(object):
    """Accepted by the constructor so code that imports it loads, but rejected as
    a scheduling strategy (not supported)."""

    def __init__(self, hard, *, soft=None):
        self.hard = hard
        self.soft = soft
