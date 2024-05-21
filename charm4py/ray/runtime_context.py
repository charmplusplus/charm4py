import charm4py

def get_runtime_context():
    return RuntimeContext(charm4py.charm.myPe())

class RuntimeContext(object):
    def __init__(self, worker):
        self.worker = worker
        self.node_id = charm4py.charm.myPe()

    @property
    def runtime_env(self):
        return {}