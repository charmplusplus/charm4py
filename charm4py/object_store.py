from charm4py import charm, Chare, Group, Array, Future, coro, Channel, Reducer
from charm4py.c_object_store import CObjectStore

class ObjectStore(Group):
    def __init__(self):
        self._object_store = CObjectStore(self.thisProxy)

    def lookup_object(self, obj_id):
        return self._object_store.lookup_object(obj_id)
    
    def lookup_location(self, obj_id):
        return self._object_store.lookup_location(obj_id)
    
    def receive_remote_object(self, obj_id, obj):
        self._object_store.receive_remote_object(obj_id, obj)

    def request_object(self, obj_id, requesting_pe):
        self._object_store.request_object(obj_id, requesting_pe)

    def request_location(self, obj_id, requesting_pe):
        self._object_store.request_location(obj_id, requesting_pe)

    def bulk_request_object(self, obj_id, requesting_pes):
        self._object_store.bulk_request_object(obj_id, requesting_pes)

    def bulk_request_location(self, obj_id, requesting_pes):
        self._object_store.bulk_request_location(obj_id, requesting_pes)
