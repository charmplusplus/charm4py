from . import placement_group as _placement_group_module
from . import scheduling_strategies
from .placement_group import (placement_group, placement_group_table, remove_placement_group,
                              get_current_placement_group, get_placement_group)
from .._private.services import get_node_ip_address
