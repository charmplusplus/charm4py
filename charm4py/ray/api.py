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
