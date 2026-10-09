"""The part of Ray's state API (``ray.util.state``) that applications use."""

from dataclasses import dataclass, field

from ... import api


@dataclass
class NodeState(object):
    node_id: str
    node_ip: str
    is_head_node: bool
    state: str
    state_message: object = None
    node_name: str = ''
    resources_total: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    start_time_ms: object = None
    end_time_ms: object = None

    def __getitem__(self, key):
        return getattr(self, key)

    def asdict(self):
        return dict(self.__dict__)


def _matches(record, filters):
    for key, op, value in filters or ():
        actual = str(getattr(record, key))
        if (op == '=' and actual != str(value)) or (op == '!=' and actual == str(value)):
            return False
    return True


def list_nodes(address=None, filters=None, limit=100, timeout=30, detail=False,
               raise_on_missing_output=True, _explain=False):
    """One NodeState per node (physical host) of the job."""
    nodes = []
    for node in api.nodes():
        res = node['Resources']
        nodes.append(NodeState(
            node_id=node['NodeID'], node_ip=node['NodeManagerAddress'],
            is_head_node='node:__internal_head__' in res,
            state='ALIVE' if node['Alive'] else 'DEAD',
            node_name=node['NodeName'],
            resources_total={k: v for k, v in res.items() if not k.startswith('node:')},
            labels=dict(node['Labels'])))
    return [n for n in nodes if _matches(n, filters)][:limit]


def get_node(id, address=None, timeout=30, _explain=False):
    for node in list_nodes():
        if node.node_id == id:
            return node
    return None
