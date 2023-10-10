from .api import get_object_store

class Future(object):
    def __init__(self):
        from ..charm import charm
        self.id = charm.get_new_future()

    def lookup_location(self):
        from ..charm import charm
        obj_store = get_object_store()
        local_obj_store = obj_store[charm.myPe()].ckLocalBranch()
        return local_obj_store.lookup_location(self.id)
    
    def lookup_object(self):
        from ..charm import charm
        obj_store = get_object_store()
        local_obj_store = obj_store[charm.myPe()].ckLocalBranch()
        return local_obj_store.lookup_object(self.id)
    
    def is_local(self):
        return self.lookup_object() != None
    
    def create_object(self, obj):
        from ..charm import charm
        obj_store = get_object_store()
        local_obj_store = obj_store[charm.myPe()].ckLocalBranch()
        local_obj_store.create_object(self.id, obj)

    def request_object(self):
        from ..charm import charm
        obj_store = get_object_store()
        obj_store[self.id % charm.numPes()].request_location_object(self.id, charm.myPe())