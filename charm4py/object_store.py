from charm4py import charm, Chare, Group, Array, Future, coro, Channel, Reducer, register
from charm4py.c_object_store import CObjectStore

_refs = None


def _holds(obj_id):
    global _refs
    if _refs is None:
        from charm4py.ray import _refs
    return _refs.holds(obj_id)

# An object is kept on a PE only while that PE holds a reference to it (see
# charm4py/ray/_refs.py): a value, location or copy that arrives after the
# object was freed -- messages are not ordered -- is dropped.

@register
class ObjectStore(Chare):
    def __init__(self):
        self._object_store = CObjectStore(self.thisProxy)

    def delete_remote_objects(self, obj_id):
        """Delete this object from the object store
        This function is called on the home pe
        """
        self._object_store.delete_remote_objects(obj_id)

    def delete_object(self, obj_id):
        """Delete this object from the local object store
        """
        self._object_store.delete_object(obj_id)

    def lookup_object(self, obj_id):
        """ Lookup object in local object map
        """
        return self._object_store.lookup_object(obj_id)
    
    def lookup_location(self, obj_id):
        """ Lookup location in local location map
        If not found in local map, send a message to home PE to get the location
        back on this PE
        """
        return self._object_store.lookup_location(obj_id)
    
    def update_location(self, obj_id, pe):
        """ Update location in local map
        Check buffers for location requests and object requests
        Also check send buffer to see if any message is buffered to send. This is
        currently not implemented, currently messages are only buffered at the 
        receiving PE
        """
        self._object_store.update_location(obj_id, pe)
        charm.check_send_buffer(obj_id)

    def register_object(self, obj_id, pe):
        """ Called on the home PE when a large object is created on ``pe``. The
        first value created for an object id wins: if a location is already
        registered (a small value stored here also registers one), this one is
        ignored. Only locations count: when ``pe`` is the home PE itself, the
        object is already in the local map before this call arrives.
        """
        if self._object_store.has_location(obj_id):
            return
        if not _holds(obj_id):
            # freed already: the copy made on ``pe`` is not needed
            self.thisProxy[pe].delete_object(obj_id)
            return
        self.update_location(obj_id, pe)

    def insert_object_small(self, obj_id, obj):
        if not _holds(obj_id):
            return   # freed already
        self._object_store.insert_object_small(obj_id, obj)

    def free_object(self, obj_id):
        """ On the home PE, once no reference to the object is left: delete it
        everywhere """
        self._object_store.free_object(obj_id)

    def num_objects(self):
        return self._object_store.num_objects()
    
    def receive_remote_object(self, obj_id, obj):
        """ Add object to local object map
        Then check receive buffer to see if any messages are buffered
        on the receiving end on this PE
        """
        if not _holds(obj_id):
            return   # no reference left here (freed, or dropped while it came)
        self._object_store.receive_remote_object(obj_id, obj)
        charm.check_receive_buffer(obj_id)
        charm.check_futures_buffer(obj_id)

    def request_object(self, obj_id, requesting_pe):
        """ If obj_id is found in the local object map, then send it back to the
        requesting PE. Else buffer the request
        """
        self._object_store.request_object(obj_id, requesting_pe)

    def request_location(self, obj_id, requesting_pe):
        """ If location for obj_id is in the local map, then send the location back to the
        requesting PE. Else buffer the request
        """
        self._object_store.request_location(obj_id, requesting_pe)

    def request_location_object(self, obj_id, requesting_pe):
        """ If location for obj_id is in the local map, send a request_location call to
        the location of obj_id and add the requesting PE to the local location map. Else
        buffer the request
        """
        self._object_store.request_location_object(obj_id, requesting_pe)

    def bulk_send_object(self, obj_id, requesting_pes):
        self._object_store.bulk_send_object(obj_id, requesting_pes)

    def bulk_send_location(self, obj_id, requesting_pes):
        self._object_store.bulk_send_location(obj_id, requesting_pes)

    def create_object(self, obj_id, obj):
        """ Add the object to the local object map and send an update_location
        call to the home PE of obj_id
        """
        self._object_store.create_object(obj_id, obj)
