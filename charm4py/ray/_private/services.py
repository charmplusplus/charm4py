from .. import api


def get_node_ip_address(address=None):
    """The routable IP of this node ('127.0.0.1' when the job has one host)."""
    return api.get_node_ip_address()
