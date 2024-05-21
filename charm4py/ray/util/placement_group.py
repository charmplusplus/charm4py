import charm4py
import charm4py.ray.api as ray


def placement_group(bundles, **kwargs):
    strategy = kwargs.pop("strategy", "PACK")
    return PlacementGroup(bundles=bundles, strategy=strategy)


def get_current_placement_group():
    return PlacementGroup()


class PlacementGroup(object):
    def __init__(self, bundles=None, strategy="PACK"):
        self.bundles = bundles
        self.strategy = strategy
        self.bundle_map = self.map_bundles(bundles, strategy)
    