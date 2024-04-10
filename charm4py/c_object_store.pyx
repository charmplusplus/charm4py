# cython: language_level=3, embedsignature=True
# distutils: language=c++

from cpython.ref cimport Py_INCREF, Py_DECREF
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdio cimport printf

import sys
import numpy as np
cimport numpy as np
cimport cython

from copy import deepcopy

cdef int OBJ_SIZE_THRESHOLD = 1024

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


cdef class MessageBuffer:
    def __init__(self):
        pass

    def __cinit__(self):
        self.dependecies = DependencyMap()

    cpdef void insert(self, object obj_ids, object msg):
        cdef int ndeps = len(obj_ids)
        cdef MessageDependency* dep = <MessageDependency*> PyMem_Malloc(
            sizeof(MessageDependency)
        )
        Py_INCREF(msg)
        deref(dep).first = <void*> msg
        deref(dep).second = ndeps
        cdef DependencyMapIterator it
        cdef DependencyList* dep_list

        cdef ObjectId obj_id
        for i in range(ndeps):
            obj_id = <ObjectId> obj_ids[i]
            it = self.dependecies.find(obj_id)
            if it == self.dependecies.end():
                self.dependecies[obj_id] = DependencyList()
            dep_list = &(self.dependecies[obj_id])
            deref(dep_list).push_back(dep)

    cpdef object check(self, ObjectId obj_id):
        cdef DependencyList* dep_list
        cdef DependencyMapIterator it = self.dependecies.find(obj_id)
        cdef DependencyListIterator dep_list_it
        cdef object completed = []
        if it != self.dependecies.end():
            dep_list = &(self.dependecies[obj_id])
            dep_list_it = deref(dep_list).begin()
            while dep_list_it != deref(dep_list).end():
                deref(dep_list_it)[0].second -= 1
                if deref(deref(dep_list_it)).second == 0:
                    # this element dependencies are satisfied
                    # send it to scheduling
                    completed.append(<object> deref(deref(dep_list_it)).first)
                    Py_DECREF(<object> deref(deref(dep_list_it)).first)
                    # remove from buffer
                    PyMem_Free(deref(dep_list_it))
                    dep_list_it = deref(dep_list).erase(dep_list_it)
                else:
                    inc(dep_list_it)
        return completed


cdef class CObjectStore:
    def __init__(self, proxy):
        self.proxy = proxy

    def __cinit__(self, proxy):
        self.replica_choice = 0
        self.object_map = ObjectMap()
        self.location_map = ObjectPEMap()
        self.obj_req_buffer = ObjectPEMap()
        self.loc_req_buffer = ObjectPEMap()
        self.obj_loc_req_buffer = ObjectPEMap()

    cpdef object lookup_object(self, ObjectId obj_id):
        #from charm4py import charm
        #charm.print_dbg("Lookup on", obj_id)
        cdef ObjectMapIterator it = self.object_map.find(obj_id)
        if it == self.object_map.end():
            return None
        return <object> deref(it).second

    cdef void insert_object(self, ObjectId obj_id, object obj):
        #FIXME when is a copy required here?
        #obj_copy = deepcopy(obj)
        if self.object_map.find(obj_id) != self.object_map.end():
            return
        Py_INCREF(obj)
        self.object_map[obj_id] = <void*> obj

    cpdef void insert_object_small(self, ObjectId obj_id, object obj):
        from charm4py import charm
        #FIXME when is a copy required here?
        #obj_copy = deepcopy(obj)
        if self.object_map.find(obj_id) != self.object_map.end():
            return
        Py_INCREF(obj)
        self.object_map[obj_id] = <void*> obj
        self.location_map[obj_id].push_back(<int> charm.myPe())
        self.check_loc_requests_buffer(obj_id)
        self.check_obj_loc_requests_buffer(obj_id)

    cpdef void delete_remote_objects(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.location_map.find(obj_id)
        if it == self.location_map.end():
            return
        cdef int* pe_arr = deref(it).second.data()
        cdef int size = deref(it).second.size()
        cdef int i
        for i in range(size):
            self.proxy[pe_arr[i]].delete_object(obj_id)

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

    cdef void buffer_obj_loc_request(self, ObjectId obj_id, int requesting_pe):
        if self.obj_loc_req_buffer.find(obj_id) == self.obj_loc_req_buffer.end():
            self.obj_loc_req_buffer[obj_id] = vector[int]()
        self.obj_loc_req_buffer[obj_id].push_back(requesting_pe)

    @cython.cdivision(True)
    cdef int choose_pe(self, vector[int] &node_list):
        # replica choice should be per entry
        cdef int pe = node_list[self.replica_choice % node_list.size()]
        self.replica_choice += 1
        return pe

    @cython.cdivision(True)
    cpdef int lookup_location(self, ObjectId obj_id, bint fetch=True):
        from charm4py import charm
        cdef ObjectPEMapIterator it = self.location_map.find(obj_id)
        if it != self.location_map.end():
            return self.choose_pe(deref(it).second)
        cdef int npes
        if fetch:
            npes = charm.numPes()
            self.proxy[obj_id % npes].request_location(obj_id, charm.myPe())
        return -1

    cdef void check_loc_requests_buffer(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.loc_req_buffer.find(obj_id)
        if it == self.loc_req_buffer.end():
            return
        # TODO is this creating a copy of the vector?
        cdef vector[int] vec = deref(it).second
        cdef np.npy_intp size = vec.size()
        cdef int pe = self.lookup_location(obj_id, fetch=False)
        cdef int[::1] arr = <int [:vec.size()]> vec.data()
        cdef np.ndarray[int, ndim=1] req_pes = np.asarray(arr)
        self.proxy[pe].bulk_request_location(obj_id, req_pes)
        self.loc_req_buffer.erase(obj_id)

    cdef void check_obj_loc_requests_buffer(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.obj_loc_req_buffer.find(obj_id)
        if it == self.obj_loc_req_buffer.end():
            return
        cdef vector[int] vec = deref(it).second
        cdef np.npy_intp size = vec.size()
        cdef int pe = self.lookup_location(obj_id, fetch=False)
        cdef int[::1] arr = <int [:vec.size()]> vec.data()
        cdef np.ndarray[int, ndim=1] req_pes = np.asarray(arr)
        self.proxy[pe].bulk_send_object(obj_id, req_pes)
        self.loc_req_buffer.erase(obj_id)

    cdef void check_obj_requests_buffer(self, ObjectId obj_id):
        cdef ObjectPEMapIterator it = self.obj_req_buffer.find(obj_id)
        if it == self.obj_req_buffer.end():
            return
        cdef vector[int] vec = deref(it).second
        cdef np.npy_intp size = vec.size()
        cdef int[::1] arr = <int [:vec.size()]> vec.data()
        cdef np.ndarray[int, ndim=1] req_pes = np.asarray(arr)
        self.bulk_send_object(obj_id, req_pes)
        self.obj_req_buffer.erase(obj_id)

    cpdef void update_location(self, ObjectId obj_id, int pe):
        cdef bint new_entry = False
        if self.location_map.find(obj_id) == self.location_map.end():
            self.location_map[obj_id] = vector[int]()
            new_entry = True
        self.location_map[obj_id].push_back(pe)
        if new_entry:
            self.check_loc_requests_buffer(obj_id)
            self.check_obj_loc_requests_buffer(obj_id)

    cpdef void request_location_object(self, ObjectId obj_id, int requesting_pe):
        # this function is intended to be called on home pes of the object id
        cdef object obj = self.lookup_object(obj_id)
        if not (obj is None):
            self.proxy[requesting_pe].receive_remote_object(obj_id, obj)
            return
        cdef int pe = self.lookup_location(obj_id, fetch=False)
        if pe == -1:
            self.buffer_obj_loc_request(obj_id, requesting_pe)
        else:
            self.proxy[pe].request_object(obj_id, requesting_pe)
            self.location_map[obj_id].push_back(requesting_pe)

    cpdef void request_location(self, ObjectId obj_id, int requesting_pe):
        # this function is intended to be called on home pes of the object id
        cdef int pe = self.lookup_location(obj_id)
        if pe == -1:
            self.buffer_loc_request(obj_id, requesting_pe)
        else:
            self.proxy[requesting_pe].update_location(obj_id, pe)

    cpdef void receive_remote_object(self, ObjectId obj_id, object obj):
        self.insert_object(obj_id, obj)
        self.check_obj_requests_buffer(obj_id)

    cpdef void request_object(self, ObjectId obj_id, int requesting_pe):
        cdef object obj = self.lookup_object(obj_id)
        if obj is None:
            self.buffer_obj_request(obj_id, requesting_pe)
        else:
            self.proxy[requesting_pe].receive_remote_object(obj_id, obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void bulk_send_object(self, ObjectId obj_id, np.ndarray[int, ndim=1] requesting_pes):
        cdef object obj = self.lookup_object(obj_id)
        for i in range(requesting_pes.shape[0]):
            self.proxy[requesting_pes[i]].receive_remote_object(obj_id, obj)

    @cython.cdivision(True)
    cpdef void create_object(self, ObjectId obj_id, object obj):
        from charm4py import charm
        cdef int npes = charm.numPes()

        # add logic to check size of obj
        cdef int size = sys.getsizeof(obj)
        if size < OBJ_SIZE_THRESHOLD:
            # in this case keep the object data on home
            self.proxy[obj_id % npes].insert_object_small(obj_id, obj)
        else:
            #insert to local object map
            self.insert_object(obj_id, obj)

            #send a message to home to add entry
            self.proxy[obj_id % npes].update_location(obj_id, charm.myPe())
