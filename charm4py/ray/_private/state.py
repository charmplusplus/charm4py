"""The subset of ``ray._private.state`` that applications use."""

from .. import api


def available_resources_per_node():
    """Node ID -> resources currently available on that node."""
    return api.available_resources_per_node()


def total_resources_per_node():
    return api.total_resources_per_node()


class GlobalState(object):

    def _available_resources_per_node(self):
        return api.available_resources_per_node()

    def _total_resources_per_node(self):
        return api.total_resources_per_node()

    def node_table(self):
        return api.nodes()

    def cluster_resources(self):
        return api.cluster_resources()

    def available_resources(self):
        return api.available_resources()

    def placement_group_table(self, placement_group_id=None):
        from ..util.placement_group import _pg_table
        return _pg_table(placement_group_id.hex() if placement_group_id is not None
                         else None)


state = GlobalState()
