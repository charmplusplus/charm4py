from charm4py import charm, Chare, Group, Array, Future, coro, Channel, Reducer, register
from charm4py.c_object_store import CObjectStore

@register
class ObjectStore(Chare):
    def __init__(self):
        self._object_store = CObjectStore(self.thisProxy)

    def lookup_object(self, obj_id):
        return self._object_store.lookup_object(obj_id)
    
    def lookup_location(self, obj_id):
        return self._object_store.lookup_location(obj_id)
    
    def update_location(self, obj_id, pe):
        self._object_store.update_location(obj_id, pe)
        charm.check_send_buffer(obj_id)
    
    def receive_remote_object(self, obj_id, obj):
        self._object_store.receive_remote_object(obj_id, obj)
        charm.check_receive_buffer(obj_id)

    def receive_remote_location(self, obj_id, pe):
        self._object_store.update_location(obj_id, pe)
        charm.check_send_buffer(obj_id)

    def request_object(self, obj_id, requesting_pe):
        self._object_store.request_object(obj_id, requesting_pe)

    def request_location(self, obj_id, requesting_pe):
        self._object_store.request_location(obj_id, requesting_pe)

    def bulk_request_object(self, obj_id, requesting_pes):
        self._object_store.bulk_request_object(obj_id, requesting_pes)

    def bulk_request_location(self, obj_id, requesting_pes):
        self._object_store.bulk_request_location(obj_id, requesting_pes)

    def create_object(self, obj_id, obj):
        self._object_store.create_object(obj_id, obj)
