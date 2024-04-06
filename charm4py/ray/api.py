import types
from copy import deepcopy

counter = 0

def init():
    from charm4py import charm, Group, ObjectStore
    global object_store
    object_store = Group(ObjectStore)
    charm.thisProxy.updateGlobals({'object_store' : object_store,},
                                  awaitable=True, module_name='charm4py.ray.api').get()


def get_object_store():
    global object_store
    return object_store

class RayProxyFunction(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call remote function without .remote()")

    def remote(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class RayProxy(object):
    def __init__(self, subclass, args, pe):
        from charm4py import Chare, register, charm
        self.proxy = Chare(subclass, args=args, onPE=pe)
        for f in dir(self.proxy):
            if not f.startswith('__'):
                setattr(self, f, RayProxyFunction(self.remote_function(f)))

    def remote_function(self, f):
        proxy_func = getattr(self.proxy, f)
        def call_remote(*args, **kwargs):
            return proxy_func(*args, **kwargs, is_ray=True)
        return call_remote


def get_ray_class(subclass):
    from charm4py import Chare, register, charm
    @register
    class RayChare(Chare):
        @staticmethod
        def remote(*a):
            global counter
            ray_proxy = RayProxy(subclass, a, counter % charm.numPes())
            counter += 1
            return ray_proxy
    return RayChare

def get_ray_task(func, num_returns):
    from charm4py import charm
    def task(*args):
        func._ck_coro = True
        result = charm.pool.map_async(func, [args], chunksize=1, multi_future=True, 
                                    num_returns=num_returns, is_ray=True)[0]
        if num_returns > 1:
            return result
        else:
            return result[0]

    return task

def remote(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # used decorator without arguments
        return remote_deco(args[0])
    else:
        num_returns = kwargs.pop("num_returns", 1)
        def wrap_remote_deco(obj):
            return remote_deco(obj, num_returns=num_returns)
        return wrap_remote_deco

def remote_deco(obj, num_returns=1):
    from charm4py import charm, Chare, register
    
    if isinstance(obj, types.FunctionType):
        obj.remote = get_ray_task(obj, num_returns)
        return obj
    else:
        # decorating without any arguments
        subclass = type(obj.__name__, (Chare, obj), {"__init__": obj.__init__})
        register(subclass)
        rayclass = get_ray_class(subclass)
        rayclass.__name__ = obj.__name__
        return rayclass


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