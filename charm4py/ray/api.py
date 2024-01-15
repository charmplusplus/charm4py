
counter = 0

def init():
    from charm4py import charm, Group, ObjectStore
    global object_store
    object_store = Group(ObjectStore)
    charm.thisProxy.updateGlobals({'object_store' : object_store},
                                  awaitable=True, module_name='charm4py.ray.api').get()


def get_object_store():
    global object_store
    return object_store


def get_ray_class(subclass):
    from charm4py import Chare, register, charm
    @register
    class RayChare(Chare):
        @staticmethod
        def remote(*a):
            global counter
            chare = Chare(subclass, args=a, onPE=counter % charm.numPes())
            counter += 1
            return chare
    return RayChare


def remote(*args, **kwargs):
    from charm4py import Chare, register
    
    if len(args) == 1 and len(kwargs) == 0:        
        # decorating without any arguments
        subclass = type(args[0].__name__, (Chare, args[0]), {"__init__": args[0].__init__})
        register(subclass)
        rayclass = get_ray_class(subclass)
        rayclass.__name__ = args[0].__name__
        return rayclass
    else:
        raise NotImplementedError("Arguments not implemented yet")
    

def get(arg):
    from charm4py import charm
    from ..threads import Future
    if isinstance(arg, Future):
        return charm.get_future_value(arg)
    elif isinstance(arg, list):
        return [charm.get_future_value(f) for f in arg]


def wait(futs, num_returns=1, timeout=None, fetch_local=True):
    # return when atleast num_returns futures have their data in the
    # local store. Similar to waitany
    if timeout != None or not fetch_local:
        raise NotImplementedError("timeout and fetch_local not implemented yet")
    from charm4py import charm
    ready = charm.getany_future_value(futs, num_returns)
    not_ready = list(set(futs) - set(ready))
    return ready, not_ready

def put(obj):
    from ..threads import Future
    from ..charm import charm
    fut = charm.threadMgr.createFuture(store=True)
    fut.create_object(obj)
    return fut