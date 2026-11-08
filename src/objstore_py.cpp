// objstore_py.cpp: nanobind port of charm4py/c_object_store.pyx and .pxd
// (SPEC-4a.md section 2.9). Defines the classes MessageBuffer and CObjectStore
// with the same method names, signatures and semantics as the Cython version.
// The main module source calls init_objstore(m) from its NB_MODULE body.
//
// No Charm++ header: the PE number and PE count come from the Python
// `charm4py.charm` object, as in the Cython code.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace {

typedef uint64_t ObjectId;

const long OBJ_SIZE_THRESHOLD = 1024;

// The `charm` object of the charm4py package. Imported on every use, as the
// Cython code did with `from charm4py import charm` inside each method
// (the module is cached in sys.modules, so this is a dictionary lookup).
nb::object charm_obj() {
  return nb::module_::import_("charm4py").attr("charm");
}

int charm_my_pe() { return nb::cast<int>(charm_obj().attr("myPe")()); }
int charm_num_pes() { return nb::cast<int>(charm_obj().attr("numPes")()); }

// A 1-D numpy.ndarray of dtype int32 holding a copy of `vec`. The array owns
// its data through a capsule (the Cython code made a view onto a local copy
// of the vector, which dangled once the method returned).
nb::object to_int32_array(const std::vector<int> &vec) {
  size_t n = vec.size();
  int32_t *data = new int32_t[n ? n : 1];
  for (size_t i = 0; i < n; i++) data[i] = (int32_t)vec[i];
  nb::capsule owner(data, [](void *p) noexcept { delete[] (int32_t *)p; });
  nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> arr(data, {n}, owner);
  return arr.cast();
}

// ---------------------------------------------------------------------------
// MessageBuffer

struct MessageDependency {
  nb::object msg;  // strong reference (Cython: Py_INCREF on a void*)
  int remaining;   // number of dependencies not yet satisfied
};
typedef std::list<std::shared_ptr<MessageDependency>> DependencyList;

class MessageBuffer {
 public:
  // insert(obj_ids, msg): obj_ids is any sequence supporting len() and
  // integer indexing (list, tuple, numpy array).
  void insert(nb::handle obj_ids, nb::handle msg) {
    int ndeps = (int)nb::len(obj_ids);
    auto dep = std::make_shared<MessageDependency>();
    dep->msg = nb::borrow(msg);
    dep->remaining = ndeps;
    for (int i = 0; i < ndeps; i++) {
      ObjectId obj_id = nb::cast<ObjectId>(obj_ids[nb::int_(i)]);
      dependencies_[obj_id].push_back(dep);
    }
  }

  // check(obj_id) -> list of messages whose dependencies are now all satisfied.
  nb::list check(ObjectId obj_id) {
    nb::list completed;
    auto it = dependencies_.find(obj_id);
    if (it != dependencies_.end()) {
      DependencyList &dep_list = it->second;
      auto dit = dep_list.begin();
      while (dit != dep_list.end()) {
        (*dit)->remaining -= 1;
        if ((*dit)->remaining == 0) {
          completed.append((*dit)->msg);
          (*dit)->msg.reset();
          dit = dep_list.erase(dit);
        } else {
          ++dit;
        }
      }
    }
    return completed;
  }

 private:
  // (spelled `dependecies` in the Cython code)
  std::unordered_map<ObjectId, DependencyList> dependencies_;
};

// ---------------------------------------------------------------------------
// CObjectStore

typedef std::unordered_map<ObjectId, nb::object> ObjectMap;
typedef std::unordered_map<ObjectId, std::vector<int>> ObjectPEMap;

class CObjectStore {
 public:
  explicit CObjectStore(nb::object proxy) : proxy_(std::move(proxy)) {}

  nb::object lookup_object(ObjectId obj_id) {
    auto it = object_map_.find(obj_id);
    if (it == object_map_.end()) return nb::none();
    return it->second;
  }

  int lookup_location(ObjectId obj_id, bool fetch) {
    auto it = location_map_.find(obj_id);
    if (it != location_map_.end()) return choose_pe(it->second);
    if (fetch) {
      int npes = charm_num_pes();
      int my_pe = charm_my_pe();
      pe_proxy((int)(obj_id % (ObjectId)npes))
          .attr("request_location")(obj_id, my_pe);
    }
    return -1;
  }

  void insert_object_small(ObjectId obj_id, nb::handle obj) {
    int my_pe = charm_my_pe();
    if (object_map_.find(obj_id) != object_map_.end()) return;
    object_map_[obj_id] = nb::borrow(obj);
    location_map_[obj_id].push_back(my_pe);
    check_loc_requests_buffer(obj_id);
    check_obj_loc_requests_buffer(obj_id);
  }

  void delete_remote_objects(ObjectId obj_id) {
    auto it = location_map_.find(obj_id);
    if (it == location_map_.end()) return;
    std::vector<int> pes = it->second;  // proxy calls may re-enter
    for (int pe : pes) pe_proxy(pe).attr("delete_object")(obj_id);
  }

  void delete_object(ObjectId obj_id) {
    auto it = object_map_.find(obj_id);
    if (it != object_map_.end()) {
      nb::object obj = std::move(it->second);
      object_map_.erase(it);
      // obj released here, after the map no longer refers to it
    }
  }

  void update_location(ObjectId obj_id, int pe) {
    bool new_entry = false;
    if (location_map_.find(obj_id) == location_map_.end()) {
      location_map_[obj_id] = std::vector<int>();
      new_entry = true;
    }
    location_map_[obj_id].push_back(pe);
    if (new_entry) {
      check_loc_requests_buffer(obj_id);
      check_obj_loc_requests_buffer(obj_id);
    }
  }

  // Intended to be called on the home PE of obj_id.
  void request_location_object(ObjectId obj_id, int requesting_pe) {
    nb::object obj = lookup_object(obj_id);
    if (!obj.is_none()) {
      pe_proxy(requesting_pe).attr("receive_remote_object")(obj_id, obj);
      return;
    }
    int pe = lookup_location(obj_id, false);
    if (pe == -1) {
      buffer_request(obj_loc_req_buffer_, obj_id, requesting_pe);
    } else {
      pe_proxy(pe).attr("request_object")(obj_id, requesting_pe);
      location_map_[obj_id].push_back(requesting_pe);
    }
  }

  // Intended to be called on the home PE of obj_id.
  void request_location(ObjectId obj_id, int requesting_pe) {
    int pe = lookup_location(obj_id, true);
    if (pe == -1) {
      buffer_request(loc_req_buffer_, obj_id, requesting_pe);
    } else {
      pe_proxy(requesting_pe).attr("update_location")(obj_id, pe);
    }
  }

  void receive_remote_object(ObjectId obj_id, nb::handle obj) {
    insert_object(obj_id, obj);
    check_obj_requests_buffer(obj_id);
  }

  void request_object(ObjectId obj_id, int requesting_pe) {
    nb::object obj = lookup_object(obj_id);
    if (obj.is_none()) {
      buffer_request(obj_req_buffer_, obj_id, requesting_pe);
    } else {
      pe_proxy(requesting_pe).attr("receive_remote_object")(obj_id, obj);
    }
  }

  // requesting_pes: 1-D sequence of PE numbers (the Cython signature required
  // a numpy int array; any sequence with len() and indexing is accepted here).
  void bulk_send_object(ObjectId obj_id, nb::handle requesting_pes) {
    nb::object obj = lookup_object(obj_id);
    size_t n = nb::len(requesting_pes);
    for (size_t i = 0; i < n; i++) {
      int pe = nb::cast<int>(requesting_pes[nb::int_(i)]);
      pe_proxy(pe).attr("receive_remote_object")(obj_id, obj);
    }
  }

  void create_object(ObjectId obj_id, nb::handle obj) {
    int npes = charm_num_pes();
    long size =
        nb::cast<long>(nb::module_::import_("sys").attr("getsizeof")(obj));
    int home = (int)(obj_id % (ObjectId)npes);
    if (size < OBJ_SIZE_THRESHOLD) {
      // keep the object data on the home PE
      pe_proxy(home).attr("insert_object_small")(obj_id, obj);
    } else {
      insert_object(obj_id, obj);
      pe_proxy(home).attr("update_location")(obj_id, charm_my_pe());
    }
  }

 private:
  friend int cobjectstore_traverse(PyObject *, visitproc, void *);
  friend int cobjectstore_clear(PyObject *);

  nb::object pe_proxy(int pe) { return proxy_[nb::int_(pe)]; }

  // cdef insert_object
  void insert_object(ObjectId obj_id, nb::handle obj) {
    if (object_map_.find(obj_id) != object_map_.end()) return;
    object_map_[obj_id] = nb::borrow(obj);
  }

  // cdef buffer_obj_request / buffer_loc_request / buffer_obj_loc_request
  static void buffer_request(ObjectPEMap &buf, ObjectId obj_id,
                             int requesting_pe) {
    buf[obj_id].push_back(requesting_pe);
  }

  // cdef choose_pe: round-robin over replicas (one counter for all entries).
  int choose_pe(const std::vector<int> &node_list) {
    int pe = node_list[replica_choice_ % node_list.size()];
    replica_choice_ += 1;
    return pe;
  }

  // cdef check_loc_requests_buffer
  void check_loc_requests_buffer(ObjectId obj_id) {
    auto it = loc_req_buffer_.find(obj_id);
    if (it == loc_req_buffer_.end()) return;
    std::vector<int> vec = it->second;
    int pe = lookup_location(obj_id, false);
    nb::object req_pes = to_int32_array(vec);
    pe_proxy(pe).attr("bulk_request_location")(obj_id, req_pes);
    loc_req_buffer_.erase(obj_id);
  }

  // cdef check_obj_loc_requests_buffer. The Cython code erases the entry of
  // loc_req_buffer here, not of obj_loc_req_buffer; kept as written.
  void check_obj_loc_requests_buffer(ObjectId obj_id) {
    auto it = obj_loc_req_buffer_.find(obj_id);
    if (it == obj_loc_req_buffer_.end()) return;
    std::vector<int> vec = it->second;
    int pe = lookup_location(obj_id, false);
    nb::object req_pes = to_int32_array(vec);
    pe_proxy(pe).attr("bulk_send_object")(obj_id, req_pes);
    loc_req_buffer_.erase(obj_id);
  }

  // cdef check_obj_requests_buffer
  void check_obj_requests_buffer(ObjectId obj_id) {
    auto it = obj_req_buffer_.find(obj_id);
    if (it == obj_req_buffer_.end()) return;
    std::vector<int> vec = it->second;
    nb::object req_pes = to_int32_array(vec);
    bulk_send_object(obj_id, req_pes);
    obj_req_buffer_.erase(obj_id);
  }

  uint64_t replica_choice_ = 0;
  ObjectMap object_map_;
  ObjectPEMap location_map_;
  ObjectPEMap obj_req_buffer_;
  ObjectPEMap loc_req_buffer_;
  ObjectPEMap obj_loc_req_buffer_;
  nb::object proxy_;
};

// Garbage-collector support for the `proxy` reference, as Cython generates for
// an `object` field (the Cython object map held void* and was not traversed;
// neither are the stored objects here).
int cobjectstore_traverse(PyObject *self, visitproc visit, void *arg) {
#if PY_VERSION_HEX >= 0x03090000
  Py_VISIT(Py_TYPE(self));
#endif
  if (!nb::inst_ready(self)) return 0;
  CObjectStore *s = nb::inst_ptr<CObjectStore>(self);
  Py_VISIT(s->proxy_.ptr());
  return 0;
}

int cobjectstore_clear(PyObject *self) {
  CObjectStore *s = nb::inst_ptr<CObjectStore>(self);
  s->proxy_ = nb::none();
  return 0;
}

PyType_Slot cobjectstore_slots[] = {
    {Py_tp_traverse, (void *)cobjectstore_traverse},
    {Py_tp_clear, (void *)cobjectstore_clear},
    {0, nullptr}};

}  // namespace

void init_objstore(nb::module_ &m) {
  nb::class_<MessageBuffer>(m, "MessageBuffer")
      .def(nb::init<>())
      .def("insert", &MessageBuffer::insert, nb::arg("obj_ids"), nb::arg("msg"))
      .def("check", &MessageBuffer::check, nb::arg("obj_id"));

  nb::class_<CObjectStore>(m, "CObjectStore",
                           nb::type_slots(cobjectstore_slots))
      .def(nb::init<nb::object>(), nb::arg("proxy"))
      .def("lookup_object", &CObjectStore::lookup_object, nb::arg("obj_id"))
      .def("lookup_location", &CObjectStore::lookup_location, nb::arg("obj_id"),
           nb::arg("fetch") = true)
      .def("insert_object_small", &CObjectStore::insert_object_small,
           nb::arg("obj_id"), nb::arg("obj"))
      .def("delete_remote_objects", &CObjectStore::delete_remote_objects,
           nb::arg("obj_id"))
      .def("delete_object", &CObjectStore::delete_object, nb::arg("obj_id"))
      .def("update_location", &CObjectStore::update_location, nb::arg("obj_id"),
           nb::arg("pe"))
      .def("request_location_object", &CObjectStore::request_location_object,
           nb::arg("obj_id"), nb::arg("requesting_pe"))
      .def("request_location", &CObjectStore::request_location,
           nb::arg("obj_id"), nb::arg("requesting_pe"))
      .def("receive_remote_object", &CObjectStore::receive_remote_object,
           nb::arg("obj_id"), nb::arg("obj"))
      .def("request_object", &CObjectStore::request_object, nb::arg("obj_id"),
           nb::arg("requesting_pe"))
      .def("bulk_send_object", &CObjectStore::bulk_send_object,
           nb::arg("obj_id"), nb::arg("requesting_pes"))
      .def("create_object", &CObjectStore::create_object, nb::arg("obj_id"),
           nb::arg("obj"));
}
