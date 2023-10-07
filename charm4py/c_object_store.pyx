# cython: language_level=3, embedsignature=True
# distutils: language=c++

from cpython.ref cimport Py_INCREF, Py_DECREF
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
cimport cython
from charm4py import myPe, numPes
from copy import deepcopy

cdef extern from "numpy/arrayobject.h":
    cdef void  import_array()

    ctypedef struct PyArrayObject:
        char  *data
        np.npy_intp *dimensions

    cdef enum NPY_TYPES:
        NPY_INT,
        NPY_UINT,
        NPY_LONG,
        NPY_FLOAT,
        NPY_DOUBLE

    np.ndarray PyArray_SimpleNewFromData(int, np.npy_intp*, int, void*)

cdef class CObjectStore:
    def __init__(self, proxy):
        self.proxy = proxy

    def __cinit__(self, proxy):
        self.replica_choice = 0
        self.object_map = ObjectMap()
        self.location_map = ObjectPEMap()
        self.obj_req_buffer = ObjectPEMap()
        self.loc_req_buffer = ObjectPEMap()

    cdef object lookup_object(self, ObjectId obj_id):
        cdef ObjectMapIterator it = self.object_map.find(obj_id)
        if it == self.object_map.end():
            return None
        return <object> deref(it).second

    cpdef void insert_object(self, ObjectId obj_id, object obj):
        #FIXME make a copy of obj before adding it to the map
        obj_copy = deepcopy(obj)
        Py_INCREF(obj_copy)
        self.object_map[obj_id] = <void*> obj_copy

    cpdef void delete_object(self, ObjectId obj_id):
        cdef ObjectMapIterator it = self.object_map.find(obj_id)
        cdef object obj
        if it != self.object_map.end():
            obj = <object> deref(it).second
            Py_DECREF(obj)
            self.object_map.erase(obj_id)

    cdef void buffer_obj_request(self, ObjectId obj_id, int requesting_pe):
        if self.obj_req_buffer.find(obj_id) == self.obj_req_buffer.end():
            self.obj_req_buffer[obj_id] = vector[int]()
        self.obj_req_buffer[obj_id].push_back(requesting_pe)

    cdef void buffer_loc_request(self, ObjectId obj_id, int requesting_pe):
        if self.loc_req_buffer.find(obj_id) == self.loc_req_buffer.end():
            self.loc_req_buffer[obj_id] = vector[int]()
        self.loc_req_buffer[obj_id].push_back(requesting_pe)

    @cython.cdivision(True)
    cdef int choose_pe(self, vector[int] &node_list):
        cdef int pe = node_list[self.replica_choice % node_list.size()]
        self.replica_choice += 1
        return pe

    cdef int lookup_location(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.location_map.find(obj_id)
        if it != self.location_map.end():
            return self.choose_pe(deref(it).second)
        return -1

    cdef void check_loc_requests_buffer(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.loc_req_buffer.find(obj_id)
        cdef int pe
        cdef np.npy_intp size
        cdef np.ndarray[int, ndim=1] req_pes
        if it != self.loc_req_buffer.end():
            size = deref(it).second.size()
            pe = self.lookup_location(obj_id)
            req_pes = PyArray_SimpleNewFromData(
                1, &size, NPY_INT, &(deref(it).second[0]))
            self.proxy[pe].bulk_request_location(obj_id, req_pes)
            self.loc_req_buffer.erase(obj_id)

    cdef void check_obj_requests_buffer(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.obj_req_buffer.find(obj_id)
        cdef int pe
        cdef np.npy_intp size
        cdef np.ndarray[int, ndim=1] req_pes
        if it != self.obj_req_buffer.end():
            size = deref(it).second.size()
            pe = self.lookup_location(obj_id)
            req_pes = PyArray_SimpleNewFromData(
                1, &size, NPY_INT, &(deref(it).second[0]))
            self.proxy[pe].bulk_request_object(obj_id, req_pes)
            self.obj_req_buffer.erase(obj_id)

    cpdef void update_location(self, ObjectId obj_id, int pe):
        cdef bint new_entry = False
        if self.location_map.find(obj_id) == self.location_map.end():
            self.location_map[obj_id] = vector[int]()
            new_entry = True
        self.location_map[obj_id].push_back(pe)
        if new_entry:
            self.check_loc_requests_buffer(obj_id)

    cpdef void request_location(self, ObjectId obj_id, int requesting_pe):
        cdef int pe = self.lookup_location(obj_id)
        if pe == -1:
            self.buffer_loc_request(obj_id, requesting_pe)
        else:
            self.proxy[pe].request_object(obj_id, requesting_pe)
            self.location_map[obj_id].push_back(requesting_pe)

    cpdef void receive_remote_object(self, ObjectId obj_id, object obj):
        self.insert_object(obj_id, obj)
        self.check_obj_requests_buffer(obj_id)

    cpdef void request_object(self, ObjectId obj_id, int requesting_pe):
        cdef object obj = self.lookup_object(obj_id)
        if obj == None:
            self.buffer_obj_request(obj_id, requesting_pe)
        else:
            self.proxy[requesting_pe].receive_remote_object(obj_id, obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void bulk_request_object(self, ObjectId obj_id, np.ndarray[int, ndim=1] requesting_pes):
        for i in range(requesting_pes.shape[0]):
            self.request_object(obj_id, requesting_pes[i])

    @cython.cdivision(True)
    cpdef void create_object(self, ObjectId obj_id, object obj):
        #insert to local object map
        self.insert_object(obj_id, obj)

        #send a message to GCS to add entry
        cdef int npes = numPes()
        self.proxy[obj_id % npes].update_location(obj_id, myPe())

        # check requests buffer
        self.check_obj_requests_buffer(obj_id)