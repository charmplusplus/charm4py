
def init():
    from charm4py import charm, Group, ObjectStore
    global object_store
    object_store = Group(ObjectStore)
    charm.thisProxy.updateGlobals({'object_store' : object_store},
                                  awaitable=True, module_name='charm4py.ray.api').get()


def get_object_store():
    global object_store
    return object_store


def remote(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0:
        # decorating without any arguments
        pass
