# cython: language_level=3, embedsignature=True
# distutils: language=c++

from libcpp.list cimport list
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

ctypedef uint64_t ObjectId
ctypedef unordered_map[ObjectId, void*] ObjectMap
ctypedef unordered_map[ObjectId, void*].iterator ObjectMapIterator
ctypedef unordered_map[ObjectId, vector[int]] ObjectPEMap
ctypedef unordered_map[ObjectId, vector[int]].iterator ObjectPEMapIterator

ctypedef pair[void*, int] MessageDependency
ctypedef list[MessageDependency*] DependencyList
ctypedef list[MessageDependency*].iterator DependencyListIterator
ctypedef unordered_map[ObjectId, DependencyList] DependencyMap
ctypedef unordered_map[ObjectId, DependencyList].iterator DependencyMapIterator


cdef class MessageBuffer:
    cdef DependencyMap dependecies

    cpdef void insert(self, object obj_ids, object msg)
    cpdef object check(self, ObjectId obj_id)


cdef class CObjectStore:
    cdef uint64_t replica_choice
    cdef ObjectMap object_map
    cdef ObjectPEMap location_map
    cdef ObjectPEMap obj_req_buffer
    cdef ObjectPEMap loc_req_buffer
    cdef ObjectPEMap obj_loc_req_buffer
    cdef object proxy

    cdef void buffer_obj_request(self, ObjectId obj_id, int requesting_pe)
    cdef void buffer_loc_request(self, ObjectId obj_id, int requesting_pe)
    cdef void buffer_obj_loc_request(self, ObjectId obj_id, int requesting_pe)
    cdef void check_obj_requests_buffer(self, ObjectId obj_id)
    cdef void check_loc_requests_buffer(self, ObjectId obj_id)
    cdef void check_obj_loc_requests_buffer(self, ObjectId obj_id)

    cpdef object lookup_object(self, ObjectId obj_id)
    cpdef int lookup_location(self, ObjectId obj_id, bint fetch=*)
    cdef void insert_object(self, ObjectId obj_id, object obj)
    cpdef void insert_object_small(self, ObjectId obj_id, object obj)
    cpdef void delete_remote_objects(self, ObjectId obj_id)
    cpdef void delete_object(self, ObjectId obj_id)

    cdef int choose_pe(self, vector[int] &node_list)

    cpdef void update_location(self, ObjectId obj_id, int pe)
    cpdef void request_location_object(self, ObjectId obj_id, int requesting_pe)
    cpdef void request_location(self, ObjectId obj_id, int requesting_pe)
    cpdef void receive_remote_object(self, ObjectId obj_id, object obj)
    cpdef void request_object(self, ObjectId obj_id, int requesting_pe)
    cpdef void bulk_send_object(self, ObjectId obj_id, np.ndarray[int, ndim=1] requesting_pes)
    cpdef void create_object(self, ObjectId obj_id, object obj)