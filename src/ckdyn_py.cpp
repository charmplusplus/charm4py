// ckdyn_py.cpp: the charm4py._ckdyn extension module, the Python binding of
// CkDyn (charm4py-thin SPEC-4a.md section 2; message convention section 1).
//
// Includes ckdyn.h only (no Charm++ header). Hooks take the GIL with
// PyGILState_Ensure; start() releases it around CkDynStart. Per-PE state is
// thread_local, so several PE threads of one reconverse process each get
// their own main greenlet and worker pool.
//
// Encodings private to this binding (both halves of the convention are in
// this file, so the Python package never sees them):
//  - a positional argument carried in the pickled tail is a BYTES slot of
//    length 0 with tag TAG_PICKLED (ckdyn.h has no way to tag a NONE slot);
//  - an array.array is a BUFFER slot with tag TAG_ARRAY | (typecode << 8)
//    (CkDynBufDesc cannot set the slot's reserved field);
//  - a PROXY slot of a proxy without an element (a broadcast proxy) has
//    pe_or_node = -1 (group kinds) or u.array.ndims = -1 - ndims (arrays).

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL CKDYN_PY_ARRAY_API
#include <numpy/arrayobject.h>

#include <greenlet/greenlet.h>

#include "ckdyn.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

// Defined in objstore_py.cpp (same module).
void init_objstore(nanobind::module_ &m);

// a ckdyn.h older than 2026-11-07 lacks this: rebuild the charm4py target of Charm++
static_assert(CKDYN_MSG_BINDING_MASK == 0xF000u, "binding flag bits 12-15");

namespace {

// ---------------------------------------------------------------------------
// Constants

enum : uint16_t {
  PYMSG_REDRESULT = 1u << 12,
  PYMSG_SINGLE    = 1u << 13,
};
enum : uint16_t { TAG_STR = 1, TAG_ARRAY = 2, TAG_PICKLED = 3 };
const int PICKLE_PROTOCOL = 5;

struct CkDynError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// Module state (SPEC 2.1)

struct MethodEntry {
  PyObject *func = nullptr;      // unbound function, NULL for an entry with no body
  PyObject *em = nullptr;
  PyObject *run = nullptr;       // em.run (strong), fetched at the first coroutine fast-path delivery
  PyObject *name = nullptr;
  uint32_t flags = 0;
  bool has_when = false;
  bool valid = false;
  int argcount = 0;
  PyObject *argnames = nullptr;
  PyObject *defaults = nullptr;
};
struct ClassEntry {
  PyObject *cls = nullptr;
  int kind = 0;
  int first_ep = -1;
  int nentries = 0;
  PyObject *proxy_class = nullptr;
};
std::vector<MethodEntry> g_methods;                 // indexed by Charm++ ep
std::vector<ClassEntry> g_classes;                  // indexed by chare index
std::unordered_map<PyTypeObject *, int> g_proxy_type_idx;
std::unordered_map<int, int> g_array_class;         // aid -> chare index (arrays created in this process)

// Sources of BUFFER slots: CkDynMsgAddBuffer records the pointer and copies the
// bytes only at CkDynMsgFinish, so each source stays alive until then.
struct KeepAlive {
  std::vector<PyObject *> objs;     // strong
  std::vector<Py_buffer *> views;   // held views (heap: a Py_buffer is not copied)
  std::vector<void *> mem;          // malloc'ed contiguous copies
  void release() {
    for (Py_buffer *v : views) { PyBuffer_Release(v); delete v; }
    views.clear();
    for (PyObject *o : objs) Py_DECREF(o);
    objs.clear();
    for (void *m : mem) free(m);
    mem.clear();
  }
  void take(KeepAlive &o) {
    objs.insert(objs.end(), o.objs.begin(), o.objs.end()); o.objs.clear();
    views.insert(views.end(), o.views.begin(), o.views.end()); o.views.clear();
    mem.insert(mem.end(), o.mem.begin(), o.mem.end()); o.mem.clear();
  }
};

struct PeState {
  PyObject *main_gr = nullptr;                      // strong
  std::vector<PyObject *> idle;                     // idle worker greenlets, strong
  int last_ep = -1;
  std::unordered_map<void *, PyObject *> packed;    // handle -> bytes (pack_size .. pack)
  PyObject *ro_packed = nullptr;
  KeepAlive reduce_keep;   // buffer sources of the last reduce hook's result (finished by CkDyn after the hook)
  // One record per worker greenlet that runs a task (key: the greenlet; no
  // reference held): the chare's target and the entry index the task runs, for
  // CkDynCoroResume/CkDynCoroSuspend around every switch into the worker. The
  // record holds no Python object, and it is erased when the task finishes.
  struct CoroRec { CkDynTarget t; int ep; uint64_t id; };
  std::unordered_map<PyObject *, CoroRec> coros;
  uint64_t next_coro_id = 1;
};
thread_local PeState tls;

PyObject *g_charm = nullptr, *g_options = nullptr;
PyObject *c_registerMainModule, *c_construct, *c_mainchare, *c_invoke, *c_process_exception,
    *c_deposit_future, *c_reduce_foreign, *c_pack, *c_unpack, *c_pack_readonlies, *c_recvReadOnly,
    *c_map_proc_num, *c_resume_from_sync, *c_destruct, *c_triggerCallable, *c_callHandler, *c_error;
PyObject *g_Future_type = nullptr, *g_remote_future = nullptr, *g_CollFuture_type = nullptr;
std::unordered_map<int, PyObject *> g_pm_name;   // ep -> proxy method name (BoundProxyMethod pickling)
PyObject *g_dumps, *g_loads, *g_array_type, *g_format_exception;
PyObject *g_DONE, *g_EXC, *g_worker_main, *g_empty_tuple;
PyObject *s_cond_next, *s_flush, *s_ckdyn_target, *s_src, *s_fid, *s_store, *s_append, *s_frombytes,
    *s_ret, *s_awaitable, *s_is_ray, *s_Future, *s_createFuture, *s_name, *s_kw_protocol_cb,
    *s_kw_buffers, *s_kw_store;
PyObject *k_block, *k_bcast, *k_bcastret, *k_creation, *k_is_ray, *k_custom_reducer, *k_single;

bool g_profiling = false, g_copy_arrays = false, g_auto_flush = true, g_in_register = false, g_started = false;
CkDynHooks g_hooks;

// ---------------------------------------------------------------------------
// Small helpers

PyObject *fetch_exc() {
#if PY_VERSION_HEX >= 0x030C0000
  return PyErr_GetRaisedException();
#else
  PyObject *t, *v, *tb;
  PyErr_Fetch(&t, &v, &tb);
  if (!t) return nullptr;
  PyErr_NormalizeException(&t, &v, &tb);
  if (tb) PyException_SetTraceback(v, tb);
  Py_XDECREF(t); Py_XDECREF(tb);
  return v;
#endif
}

void restore_exc(PyObject *exc) {   // steals exc
#if PY_VERSION_HEX >= 0x030C0000
  PyErr_SetRaisedException(exc);
#else
  PyErr_Restore(Py_NewRef((PyObject *)Py_TYPE(exc)), exc, PyException_GetTraceback(exc));
#endif
}

// Call with borrowed positional arguments; new reference or NULL.
PyObject *call(PyObject *fn, std::initializer_list<PyObject *> a) {
  PyObject *buf[8];
  size_t n = 0;
  for (PyObject *x : a) buf[n++] = x;
  return PyObject_Vectorcall(fn, buf, n, nullptr);
}

void flush_stdio() {
  for (const char *name : {"stdout", "stderr"}) {
    PyObject *f = PySys_GetObject(name);   // borrowed
    if (f && f != Py_None) {
      PyObject *r = PyObject_CallMethod(f, "flush", nullptr);
      if (r) Py_DECREF(r); else PyErr_Clear();
    }
  }
}

void flush_at_exit() {
  if (!Py_IsInitialized()) return;
  PyGILState_STATE s = PyGILState_Ensure();
  flush_stdio();
  PyGILState_Release(s);
}

std::string exception_text() {
  PyObject *exc = fetch_exc();
  if (!exc) return "(no Python exception set)\n";
  std::string out;
  PyObject *lines = g_format_exception ? call(g_format_exception, {exc}) : nullptr;
  if (lines) {
    PyObject *sep = PyUnicode_FromString("");
    PyObject *joined = PyUnicode_Join(sep, lines);
    Py_DECREF(sep); Py_DECREF(lines);
    if (joined) { const char *u = PyUnicode_AsUTF8(joined); if (u) out = u; Py_DECREF(joined); }
  }
  if (out.empty()) {
    PyErr_Clear();
    PyObject *s = PyObject_Repr(exc);
    if (s) { const char *u = PyUnicode_AsUTF8(s); if (u) out = std::string(u) + "\n"; Py_DECREF(s); }
  }
  PyErr_Clear();
  Py_DECREF(exc);
  return out;
}

// An exception escaped a hook: report it through charm._error and abort.
[[noreturn]] void hook_abort(const char *where) {
  std::string text = std::string("charm4py._ckdyn: exception in ") + where + ":\n" + exception_text();
  bool reported = false;
  if (c_error) {
    PyObject *s = PyUnicode_FromString(text.c_str());
    PyObject *r = s ? call(c_error, {s}) : nullptr;
    Py_XDECREF(s);
    if (r) { Py_DECREF(r); reported = true; } else PyErr_Clear();
  }
  if (!reported) { fputs(text.c_str(), stderr); fflush(stderr); }
  flush_stdio();
  CkDynAbort(text.c_str());
  abort();
}

void cpp_exc_to_python() {
  try { throw; }
  catch (nb::python_error &e) { e.restore(); }
  catch (CkDynError &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); }
  catch (std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); }
  catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception"); }
}

PyObject *charm4py_error_type() {
  PyObject *mod = PyImport_ImportModule("charm4py.charm");
  if (mod) {
    PyObject *t = PyObject_GetAttrString(mod, "Charm4PyError");
    Py_DECREF(mod);
    if (t) return t;
  }
  PyErr_Clear();
  return Py_NewRef(PyExc_RuntimeError);
}

[[noreturn]] void throw_py() { throw nb::python_error(); }

struct Gil {
  PyGILState_STATE s;
  Gil() : s(PyGILState_Ensure()) {
    if (!tls.main_gr) tls.main_gr = (PyObject *)PyGreenlet_GetCurrent();
  }
  ~Gil() { PyGILState_Release(s); }
};

// Owned list of PyObject* with inline storage.
struct ObjVec {
  PyObject *inl[8];
  PyObject **p = inl;
  size_t n = 0, cap = 8;
  ObjVec() = default;
  ObjVec(const ObjVec &) = delete;
  ~ObjVec() {
    for (size_t i = 0; i < n; i++) Py_XDECREF(p[i]);
    if (p != inl) free(p);
  }
  void push(PyObject *o) {
    if (n == cap) {
      size_t nc = cap * 2;
      PyObject **np_ = (PyObject **)malloc(nc * sizeof(PyObject *));
      memcpy(np_, p, n * sizeof(PyObject *));
      if (p != inl) free(p);
      p = np_; cap = nc;
    }
    p[n++] = o;
  }
  PyObject *list() const {   // new list with new references
    PyObject *l = PyList_New((Py_ssize_t)n);
    if (!l) return nullptr;
    for (size_t i = 0; i < n; i++) PyList_SET_ITEM(l, (Py_ssize_t)i, Py_NewRef(p[i]));
    return l;
  }
};

size_t elem_size(int elem) {
  switch (elem) {
    case CKDYN_ELEM_I16: case CKDYN_ELEM_U16: case CKDYN_ELEM_F16: return 2;
    case CKDYN_ELEM_I32: case CKDYN_ELEM_U32: case CKDYN_ELEM_F32: return 4;
    case CKDYN_ELEM_I64: case CKDYN_ELEM_U64: case CKDYN_ELEM_F64: case CKDYN_ELEM_C64: return 8;
    case CKDYN_ELEM_C128: return 16;
    default: return 1;
  }
}

int elem_typenum(int elem) {
  switch (elem) {
    case CKDYN_ELEM_I8: return NPY_INT8;
    case CKDYN_ELEM_I16: return NPY_INT16;
    case CKDYN_ELEM_I32: return NPY_INT32;
    case CKDYN_ELEM_I64: return NPY_INT64;
    case CKDYN_ELEM_U8: return NPY_UINT8;
    case CKDYN_ELEM_U16: return NPY_UINT16;
    case CKDYN_ELEM_U32: return NPY_UINT32;
    case CKDYN_ELEM_U64: return NPY_UINT64;
    case CKDYN_ELEM_F16: return NPY_FLOAT16;
    case CKDYN_ELEM_F32: return NPY_FLOAT32;
    case CKDYN_ELEM_F64: return NPY_FLOAT64;
    case CKDYN_ELEM_C64: return NPY_COMPLEX64;
    case CKDYN_ELEM_C128: return NPY_COMPLEX128;
    case CKDYN_ELEM_BOOL: return NPY_BOOL;
    default: return NPY_UINT8;
  }
}

int np_elem(PyArrayObject *a) {
  if (!PyArray_ISNOTSWAPPED(a)) return -1;
  PyArray_Descr *d = PyArray_DESCR(a);
  npy_intp sz = PyArray_ITEMSIZE(a);
  switch (d->kind) {
    case 'b': return sz == 1 ? CKDYN_ELEM_BOOL : -1;
    case 'i': return sz == 1 ? CKDYN_ELEM_I8 : sz == 2 ? CKDYN_ELEM_I16 : sz == 4 ? CKDYN_ELEM_I32 : sz == 8 ? CKDYN_ELEM_I64 : -1;
    case 'u': return sz == 1 ? CKDYN_ELEM_U8 : sz == 2 ? CKDYN_ELEM_U16 : sz == 4 ? CKDYN_ELEM_U32 : sz == 8 ? CKDYN_ELEM_U64 : -1;
    case 'f': return sz == 2 ? CKDYN_ELEM_F16 : sz == 4 ? CKDYN_ELEM_F32 : sz == 8 ? CKDYN_ELEM_F64 : -1;
    case 'c': return sz == 8 ? CKDYN_ELEM_C64 : sz == 16 ? CKDYN_ELEM_C128 : -1;
    default: return -1;
  }
}

int typecode_elem(char tc, Py_ssize_t itemsize) {
  switch (tc) {
    case 'b': return CKDYN_ELEM_I8;
    case 'B': return CKDYN_ELEM_U8;
    case 'h': return CKDYN_ELEM_I16;
    case 'H': return CKDYN_ELEM_U16;
    case 'i': case 'l': case 'q':
      return itemsize == 4 ? CKDYN_ELEM_I32 : itemsize == 8 ? CKDYN_ELEM_I64 : -1;
    case 'I': case 'L': case 'Q':
      return itemsize == 4 ? CKDYN_ELEM_U32 : itemsize == 8 ? CKDYN_ELEM_U64 : -1;
    case 'f': return CKDYN_ELEM_F32;
    case 'd': return CKDYN_ELEM_F64;
    default: return -1;
  }
}

// ---------------------------------------------------------------------------
// Target (nanobind class)

struct TargetW {
  CkDynTarget t;
};

PyObject *make_target(const CkDynTarget &t) {
  return nb::cast(TargetW{t}).release().ptr();
}

bool target_eq(const CkDynTarget &a, const CkDynTarget &b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case CKDYN_MAINCHARE: case CKDYN_CHARE:
      return a.u.chare.pe == b.u.chare.pe && a.u.chare.ptr == b.u.chare.ptr;
    case CKDYN_ARRAY:
      if (a.u.array.aid != b.u.array.aid || a.u.array.ndims != b.u.array.ndims) return false;
      for (int i = 0; i < a.u.array.ndims && i < CKDYN_MAX_DIMS; i++)
        if (a.u.array.index[i] != b.u.array.index[i]) return false;
      return true;
    default:
      return a.u.group.gid == b.u.group.gid && a.u.group.pe_or_node == b.u.group.pe_or_node;
  }
}

Py_hash_t target_hash(const CkDynTarget &t) {
  uint64_t h = 1469598103934665603ull ^ (uint64_t)t.kind;
  auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  switch (t.kind) {
    case CKDYN_MAINCHARE: case CKDYN_CHARE: mix((uint64_t)t.u.chare.pe); mix((uint64_t)(uintptr_t)t.u.chare.ptr); break;
    case CKDYN_ARRAY:
      mix((uint64_t)t.u.array.aid); mix((uint64_t)t.u.array.ndims);
      for (int i = 0; i < t.u.array.ndims && i < CKDYN_MAX_DIMS; i++) mix((uint64_t)(int64_t)t.u.array.index[i]);
      break;
    default: mix((uint64_t)t.u.group.gid); mix((uint64_t)(int64_t)t.u.group.pe_or_node); break;
  }
  Py_hash_t r = (Py_hash_t)h;
  return r == -1 ? -2 : r;
}

PyObject *index_tuple(const int *index, int ndims) {
  PyObject *t = PyTuple_New(ndims);
  for (int i = 0; i < ndims; i++) PyTuple_SET_ITEM(t, i, PyLong_FromLong(index[i]));
  return t;
}

CkDynTarget target_of(PyObject *obj) {
  PyObject *a = PyObject_GetAttr(obj, s_ckdyn_target);
  if (!a) throw_py();
  TargetW *tw = nullptr;
  bool ok = nb::try_cast<TargetW *>(nb::handle(a), tw, false) && tw;
  CkDynTarget t;
  if (ok) t = tw->t;
  Py_DECREF(a);
  if (!ok) throw nb::type_error("_ckdyn_target is not a _ckdyn.Target");
  return t;
}

// ---------------------------------------------------------------------------
// MsgRef: holds one CkDynMsgRetain of a delivered message

struct MsgRefObject {
  PyObject_HEAD
  void *msg;
};

void msgref_dealloc(PyObject *self) {
  MsgRefObject *m = (MsgRefObject *)self;
  if (m->msg) CkDynMsgRelease(m->msg);
  Py_TYPE(self)->tp_free(self);
}

PyTypeObject MsgRefType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyObject *msgref_new(void *msg) {
  MsgRefObject *m = PyObject_New(MsgRefObject, &MsgRefType);
  if (m) m->msg = msg;
  return (PyObject *)m;
}

// ---------------------------------------------------------------------------
// Proxy (raw C type, base of every generated proxy class)

struct ProxyObject {
  PyObject_HEAD
  CkDynTarget t;
  int has_elem;
};

PyTypeObject ProxyType = {PyVarObject_HEAD_INIT(nullptr, 0)};

inline bool is_proxy(PyObject *o) { return PyObject_TypeCheck(o, &ProxyType); }
inline bool is_group_kind(int k) { return k == CKDYN_GROUP || k == CKDYN_NODEGROUP || k == CKDYN_ARRAYMAP; }

int proxy_init(PyObject *self, PyObject *args, PyObject *kwds) {
  ProxyObject *p = (ProxyObject *)self;
  Py_ssize_t n = PyTuple_GET_SIZE(args);
  if (n == 1) {   // Proxy(target)
    TargetW *tw = nullptr;
    if (!nb::try_cast<TargetW *>(nb::handle(PyTuple_GET_ITEM(args, 0)), tw, false) || !tw) {
      PyErr_SetString(PyExc_TypeError, "Proxy(target): target must be a _ckdyn.Target");
      return -1;
    }
    p->t = tw->t;
    p->has_elem = 1;
    return 0;
  }
  int kind, id, ndims = 0, chare_idx = -1;
  if (!PyArg_ParseTuple(args, "ii|ii", &kind, &id, &ndims, &chare_idx)) return -1;
  memset(&p->t, 0, sizeof p->t);
  if (chare_idx < 0) {
    auto it = g_proxy_type_idx.find(Py_TYPE(self));
    if (it != g_proxy_type_idx.end()) chare_idx = it->second;
  }
  p->t.kind = (CkDynKind)kind;
  p->t.chare_idx = chare_idx;
  if (is_group_kind(kind)) {
    p->t.u.group.gid = id;
    p->t.u.group.pe_or_node = -1;
  } else if (kind == CKDYN_ARRAY) {
    if (ndims < 0 || ndims > CKDYN_MAX_DIMS) { PyErr_SetString(PyExc_ValueError, "array proxy: ndims out of range"); return -1; }
    p->t.u.array.aid = id;
    p->t.u.array.ndims = ndims;
  } else {
    PyErr_SetString(PyExc_ValueError, "Proxy(kind, id, ndims): kind must be GROUP, NODEGROUP, ARRAYMAP or ARRAY (chares: Proxy._from_target)");
    return -1;
  }
  p->has_elem = 0;
  return 0;
}

// A new instance of cls (a Proxy subclass) without calling __init__.
PyObject *proxy_alloc(PyObject *cls, const CkDynTarget &t, bool has_elem) {
  if (!PyType_Check(cls) || !PyType_IsSubtype((PyTypeObject *)cls, &ProxyType)) {
    PyErr_SetString(PyExc_TypeError, "not a _ckdyn.Proxy subclass");
    return nullptr;
  }
  PyTypeObject *tp = (PyTypeObject *)cls;
  PyObject *o = tp->tp_new(tp, g_empty_tuple, nullptr);
  if (!o) return nullptr;
  if (!is_proxy(o)) { Py_DECREF(o); PyErr_SetString(PyExc_TypeError, "__new__ did not return a Proxy"); return nullptr; }
  ProxyObject *p = (ProxyObject *)o;
  p->t = t;
  p->has_elem = has_elem ? 1 : 0;
  return o;
}

PyObject *proxy_from_target(PyObject *cls, PyObject *arg) {
  TargetW *tw = nullptr;
  if (!nb::try_cast<TargetW *>(nb::handle(arg), tw, false) || !tw) {
    PyErr_SetString(PyExc_TypeError, "_from_target: argument must be a _ckdyn.Target");
    return nullptr;
  }
  bool has = true;
  if (is_group_kind(tw->t.kind)) has = tw->t.u.group.pe_or_node >= 0;
  return proxy_alloc(cls, tw->t, has);
}

PyObject *proxy_get_gid(PyObject *self, void *) {
  ProxyObject *p = (ProxyObject *)self;
  if (!is_group_kind(p->t.kind)) { PyErr_SetString(PyExc_AttributeError, "gid"); return nullptr; }
  return PyLong_FromLong(p->t.u.group.gid);
}
PyObject *proxy_get_aid(PyObject *self, void *) {
  ProxyObject *p = (ProxyObject *)self;
  if (p->t.kind != CKDYN_ARRAY) { PyErr_SetString(PyExc_AttributeError, "aid"); return nullptr; }
  return PyLong_FromLong(p->t.u.array.aid);
}
PyObject *proxy_get_cid(PyObject *self, void *) {
  ProxyObject *p = (ProxyObject *)self;
  if (p->t.kind != CKDYN_CHARE && p->t.kind != CKDYN_MAINCHARE) { PyErr_SetString(PyExc_AttributeError, "cid"); return nullptr; }
  return Py_BuildValue("(iK)", p->t.u.chare.pe, (unsigned long long)(uintptr_t)p->t.u.chare.ptr);
}
PyObject *proxy_get_ndims(PyObject *self, void *) {
  ProxyObject *p = (ProxyObject *)self;
  if (p->t.kind != CKDYN_ARRAY) { PyErr_SetString(PyExc_AttributeError, "ndims"); return nullptr; }
  return PyLong_FromLong(p->t.u.array.ndims);
}
PyObject *proxy_get_elemIdx(PyObject *self, void *) {
  ProxyObject *p = (ProxyObject *)self;
  if (is_group_kind(p->t.kind)) return PyLong_FromLong(p->has_elem ? p->t.u.group.pe_or_node : -1);
  if (p->t.kind == CKDYN_ARRAY) return p->has_elem ? index_tuple(p->t.u.array.index, p->t.u.array.ndims) : PyTuple_New(0);
  Py_RETURN_NONE;
}
PyObject *proxy_get_issec(PyObject *, void *) { Py_RETURN_FALSE; }
PyObject *proxy_get_section(PyObject *, void *) { Py_RETURN_NONE; }
PyObject *proxy_get_target(PyObject *self, void *) { return make_target(((ProxyObject *)self)->t); }
PyObject *proxy_get_has_elem(PyObject *self, void *) { return PyBool_FromLong(((ProxyObject *)self)->has_elem); }
PyObject *proxy_get_kind(PyObject *self, void *) { return PyLong_FromLong(((ProxyObject *)self)->t.kind); }
PyObject *proxy_get_chare_idx(PyObject *self, void *) { return PyLong_FromLong(((ProxyObject *)self)->t.chare_idx); }

PyObject *proxy_set_elem(PyObject *self, PyObject *arg) {
  ProxyObject *p = (ProxyObject *)self;
  if (!is_group_kind(p->t.kind)) { PyErr_SetString(PyExc_TypeError, "_set_elem on a proxy that is not a group proxy"); return nullptr; }
  long pe = PyLong_AsLong(arg);
  if (pe == -1 && PyErr_Occurred()) return nullptr;
  p->t.u.group.pe_or_node = (int)pe;
  p->has_elem = pe >= 0;
  Py_RETURN_NONE;
}

PyObject *proxy_set_index(PyObject *self, PyObject *arg) {
  ProxyObject *p = (ProxyObject *)self;
  if (p->t.kind != CKDYN_ARRAY) { PyErr_SetString(PyExc_TypeError, "_set_index on a proxy that is not an array proxy"); return nullptr; }
  PyObject *seq = PySequence_Fast(arg, "_set_index: index must be a tuple of ints");
  if (!seq) return nullptr;
  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n < 1 || n > CKDYN_MAX_DIMS) { Py_DECREF(seq); PyErr_SetString(PyExc_ValueError, "array index must have 1 to 6 dimensions"); return nullptr; }
  int idx[CKDYN_MAX_DIMS] = {0};
  for (Py_ssize_t i = 0; i < n; i++) {
    long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
    if (v == -1 && PyErr_Occurred()) { Py_DECREF(seq); return nullptr; }
    idx[i] = (int)v;
  }
  Py_DECREF(seq);
  p->t.u.array.ndims = (int)n;
  memcpy(p->t.u.array.index, idx, sizeof idx);
  p->has_elem = 1;
  Py_RETURN_NONE;
}

PyObject *proxy_clear_elem(PyObject *self, PyObject *) {
  ProxyObject *p = (ProxyObject *)self;
  if (is_group_kind(p->t.kind)) p->t.u.group.pe_or_node = -1;
  else if (p->t.kind == CKDYN_ARRAY) memset(p->t.u.array.index, 0, sizeof p->t.u.array.index);
  else { PyErr_SetString(PyExc_TypeError, "_clear_elem on a chare proxy"); return nullptr; }
  p->has_elem = 0;
  Py_RETURN_NONE;
}

PyObject *proxy_getstate(PyObject *self, PyObject *) {
  ProxyObject *p = (ProxyObject *)self;
  return Py_BuildValue("(y#i)", (const char *)&p->t, (Py_ssize_t)sizeof p->t, p->has_elem);
}

PyObject *proxy_setstate(PyObject *self, PyObject *state) {
  ProxyObject *p = (ProxyObject *)self;
  const char *buf; Py_ssize_t len; int has;
  if (!PyArg_ParseTuple(state, "y#i", &buf, &len, &has)) return nullptr;
  if (len != (Py_ssize_t)sizeof p->t) { PyErr_SetString(PyExc_ValueError, "Proxy state of the wrong size"); return nullptr; }
  memcpy(&p->t, buf, sizeof p->t);
  p->has_elem = has;
  Py_RETURN_NONE;
}

PyObject *proxy_richcompare(PyObject *a, PyObject *b, int op) {
  if ((op != Py_EQ && op != Py_NE) || !is_proxy(b)) Py_RETURN_NOTIMPLEMENTED;
  ProxyObject *x = (ProxyObject *)a, *y = (ProxyObject *)b;
  bool eq = x->has_elem == y->has_elem && target_eq(x->t, y->t);
  if (eq == (op == Py_EQ)) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

Py_hash_t proxy_hash(PyObject *self) {
  ProxyObject *p = (ProxyObject *)self;
  Py_hash_t h = target_hash(p->t) ^ (Py_hash_t)p->has_elem;
  return h == -1 ? -2 : h;
}

PyObject *proxy_repr(PyObject *self) {
  ProxyObject *p = (ProxyObject *)self;
  PyObject *e = proxy_get_elemIdx(self, nullptr);
  PyObject *r = PyUnicode_FromFormat("<%s kind=%d chare_idx=%d elem=%R>", Py_TYPE(self)->tp_name, (int)p->t.kind, p->t.chare_idx, e);
  Py_XDECREF(e);
  return r;
}

PyGetSetDef proxy_getset[] = {
    {"gid", proxy_get_gid, nullptr, nullptr, nullptr},
    {"aid", proxy_get_aid, nullptr, nullptr, nullptr},
    {"cid", proxy_get_cid, nullptr, nullptr, nullptr},
    {"ndims", proxy_get_ndims, nullptr, nullptr, nullptr},
    {"elemIdx", proxy_get_elemIdx, nullptr, nullptr, nullptr},
    {"issec", proxy_get_issec, nullptr, nullptr, nullptr},
    {"section", proxy_get_section, nullptr, nullptr, nullptr},
    {"_target", proxy_get_target, nullptr, nullptr, nullptr},
    {"_has_elem", proxy_get_has_elem, nullptr, nullptr, nullptr},
    {"_kind", proxy_get_kind, nullptr, nullptr, nullptr},
    {"_chare_idx", proxy_get_chare_idx, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyMethodDef proxy_methods[] = {
    {"_set_elem", (PyCFunction)proxy_set_elem, METH_O, nullptr},
    {"_set_index", (PyCFunction)proxy_set_index, METH_O, nullptr},
    {"_clear_elem", (PyCFunction)proxy_clear_elem, METH_NOARGS, nullptr},
    {"__getstate__", (PyCFunction)proxy_getstate, METH_NOARGS, nullptr},
    {"__setstate__", (PyCFunction)proxy_setstate, METH_O, nullptr},
    {"_from_target", (PyCFunction)proxy_from_target, METH_O | METH_CLASS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

// ---------------------------------------------------------------------------
// Callback (nanobind class)

struct CallbackW {
  CkDynCallback cb;
};

// ---------------------------------------------------------------------------
// ProxyMethod and BoundProxyMethod (raw C types)

struct PMObject {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  int ep;
  int argcount;
  PyObject *argnames;   // tuple
  PyObject *defaults;   // tuple
  PyObject *name;
  PyObject *qualname;
};

struct BPMObject {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  PyObject *proxy;
  PMObject *pm;
};

PyTypeObject PMType = {PyVarObject_HEAD_INIT(nullptr, 0)};
PyTypeObject BPMType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// ---------------------------------------------------------------------------
// Sender side of the message convention (SPEC section 1)

struct BuilderGuard {
  CkDynMsgBuilder *b;
  explicit BuilderGuard(CkDynMsgBuilder *b_) : b(b_) {}
  ~BuilderGuard() { if (b) CkDynMsgAbandon(b); }
  CkDynMsg *finish(int ep) { CkDynMsg *m = CkDynMsgFinish(b, ep); b = nullptr; return m; }
};

bool future_ids(PyObject *f, int *pe, uint32_t *fid) {
  PyObject *s = PyObject_GetAttr(f, s_src);
  if (!s) return false;
  long v = PyLong_AsLong(s);
  Py_DECREF(s);
  if (v == -1 && PyErr_Occurred()) return false;
  PyObject *i = PyObject_GetAttr(f, s_fid);
  if (!i) return false;
  unsigned long long w = PyLong_AsUnsignedLongLong(i);
  Py_DECREF(i);
  if (w == (unsigned long long)-1 && PyErr_Occurred()) return false;
  if (w > 0xffffffffull) { PyErr_SetString(PyExc_OverflowError, "future id does not fit in 32 bits"); return false; }
  *pe = (int)v; *fid = (uint32_t)w;
  return true;
}

struct Packer {
  CkDynMsgBuilder *b;
  PyObject *pickled = nullptr;   // list of values for the tail
  KeepAlive keep;                // BUFFER sources: the Packer must outlive CkDynMsgFinish/Abandon
  explicit Packer(CkDynMsgBuilder *b_) : b(b_) {}
  ~Packer() { Py_XDECREF(pickled); keep.release(); }

  bool pickle(PyObject *o) {
    if (!pickled && !(pickled = PyList_New(0))) return false;
    if (PyList_Append(pickled, o) < 0) return false;
    CkDynMsgAddBytes(b, nullptr, 0, TAG_PICKLED);
    return true;
  }

  bool add_ndarray(PyObject *o) {
    PyArrayObject *a = (PyArrayObject *)o;
    int elem = np_elem(a);
    int ndim = PyArray_NDIM(a);
    if (elem < 0 || ndim > CKDYN_MAX_BUF_DIMS) return pickle(o);
    PyArrayObject *c = a;
    bool fortran = false;
    if (PyArray_IS_C_CONTIGUOUS(a)) {
    } else if (PyArray_IS_F_CONTIGUOUS(a)) {
      fortran = true;
    } else {
      c = PyArray_GETCONTIGUOUS(a);
      if (!c) return false;
    }
    npy_intp nbytes = PyArray_NBYTES(c);
    if ((uint64_t)nbytes > 0xffffffffull) {
      if (c != a) Py_DECREF(c);
      PyErr_SetString(PyExc_ValueError, "numpy array argument of 4 GB or more");
      return false;
    }
    int64_t shape[CKDYN_MAX_BUF_DIMS];
    for (int i = 0; i < ndim; i++) shape[i] = (int64_t)PyArray_DIM(c, i);
    CkDynBufDesc d;
    memset(&d, 0, sizeof d);
    d.ptr = PyArray_DATA(c);
    d.len = (uint64_t)nbytes;
    d.elem = (uint8_t)elem;
    d.ndim = (uint8_t)ndim;
    d.bflags = fortran ? CKDYN_BUF_FORTRAN : 0;
    d.tag = 0;
    d.shape = ndim ? shape : nullptr;
    d.strides = nullptr;
    CkDynMsgAddBuffer(b, &d);
    keep.objs.push_back(c != a ? (PyObject *)c : Py_NewRef(o));   // c's reference moves to keep
    return true;
  }

  bool add_array(PyObject *o) {   // array.array
    Py_buffer *vp = new Py_buffer;
    Py_buffer &v = *vp;
    if (PyObject_GetBuffer(o, &v, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) { delete vp; return false; }
    char tc = v.format ? v.format[0] : 0;
    int elem = typecode_elem(tc, v.itemsize);
    if (elem < 0 || (uint64_t)v.len > 0xffffffffull) { PyBuffer_Release(&v); delete vp; return pickle(o); }
    int64_t shape[1] = {(int64_t)(v.itemsize ? v.len / v.itemsize : 0)};
    CkDynBufDesc d;
    memset(&d, 0, sizeof d);
    d.ptr = v.buf; d.len = (uint64_t)v.len; d.elem = (uint8_t)elem; d.ndim = 1;
    d.tag = (uint16_t)(TAG_ARRAY | ((uint16_t)(unsigned char)tc << 8));
    d.shape = shape;
    CkDynMsgAddBuffer(b, &d);
    keep.views.push_back(vp);   // released after Finish
    return true;
  }

  bool add(PyObject *o) {
    if (o == Py_None) { CkDynMsgAddNone(b); return true; }
    PyTypeObject *tp = Py_TYPE(o);
    if (tp == &PyBool_Type) { CkDynMsgAddBool(b, o == Py_True); return true; }
    if (tp == &PyLong_Type) {
      int ovf = 0;
      long long v = PyLong_AsLongLongAndOverflow(o, &ovf);
      if (v == -1 && PyErr_Occurred()) return false;
      if (!ovf) { CkDynMsgAddI64(b, (int64_t)v); return true; }
      if (ovf > 0) {
        unsigned long long u = PyLong_AsUnsignedLongLong(o);
        if (!(u == (unsigned long long)-1 && PyErr_Occurred())) { CkDynMsgAddU64(b, (uint64_t)u); return true; }
        PyErr_Clear();
      }
      return pickle(o);
    }
    if (tp == &PyFloat_Type) { CkDynMsgAddF64(b, PyFloat_AS_DOUBLE(o)); return true; }
    if (tp == &PyBytes_Type) {
      Py_ssize_t n = PyBytes_GET_SIZE(o);
      if ((uint64_t)n > 0xffffffffull) return pickle(o);
      CkDynMsgAddBytes(b, PyBytes_AS_STRING(o), (uint32_t)n, 0);
      return true;
    }
    if (tp == &PyUnicode_Type) {
      Py_ssize_t n;
      const char *s = PyUnicode_AsUTF8AndSize(o, &n);
      if (!s) return false;
      if ((uint64_t)n > 0xffffffffull) return pickle(o);
      CkDynMsgAddBytes(b, s, (uint32_t)n, TAG_STR);
      return true;
    }
    if (tp == &PyArray_Type) return add_ndarray(o);
    if ((PyObject *)tp == g_array_type) return add_array(o);
    if ((PyObject *)tp == g_Future_type) {
      PyObject *st = PyObject_GetAttr(o, s_store);
      if (!st) return false;
      int store = PyObject_IsTrue(st);
      Py_DECREF(st);
      if (store < 0) return false;
      if (store) return pickle(o);
      int pe; uint32_t fid;
      if (!future_ids(o, &pe, &fid)) return false;
      CkDynMsgAddFuture(b, pe, fid);
      return true;
    }
    if (PyObject_TypeCheck(o, &ProxyType)) {
      ProxyObject *p = (ProxyObject *)o;
      CkDynTarget t = p->t;
      if (!p->has_elem) {
        if (t.kind == CKDYN_ARRAY) t.u.array.ndims = -1 - t.u.array.ndims;
        else if (is_group_kind(t.kind)) t.u.group.pe_or_node = -1;
      }
      CkDynMsgAddProxy(b, &t);
      return true;
    }
    return pickle(o);
  }

  bool add_oob(PyObject *pb) {
    Py_buffer *vp = new Py_buffer;
    Py_buffer &v = *vp;
    if (PyObject_GetBuffer(pb, &v, PyBUF_FULL_RO) < 0) { delete vp; return false; }
    keep.views.push_back(vp);   // released after Finish (or by the Packer on an error)
    if ((uint64_t)v.len > 0xffffffffull) {
      PyErr_SetString(PyExc_ValueError, "pickled buffer of 4 GB or more");
      return false;
    }
    const void *ptr = v.buf;
    if (!PyBuffer_IsContiguous(&v, 'A')) {
      void *tmp = malloc(v.len ? v.len : 1);
      keep.mem.push_back(tmp);
      if (PyBuffer_ToContiguous(tmp, &v, v.len, 'A') < 0) return false;
      ptr = tmp;
    }
    int64_t shape[1] = {(int64_t)v.len};
    CkDynBufDesc d;
    memset(&d, 0, sizeof d);
    d.ptr = ptr; d.len = (uint64_t)v.len; d.elem = CKDYN_ELEM_BYTE; d.ndim = 1;
    d.bflags = CKDYN_BUF_OOB; d.shape = shape;
    CkDynMsgAddBuffer(b, &d);
    return true;
  }

  // OOB buffers and the tail, after the positional slots.
  bool finish() {
    if (!pickled) return true;
    PyObject *tup = PyList_AsTuple(pickled);
    if (!tup) return false;
    PyObject *oob = PyList_New(0);
    PyObject *cb = oob ? PyObject_GetAttr(oob, s_append) : nullptr;
    PyObject *proto = PyLong_FromLong(PICKLE_PROTOCOL);
    PyObject *data = nullptr;
    if (cb && proto) {
      PyObject *argv[3] = {tup, proto, cb};
      data = PyObject_Vectorcall(g_dumps, argv, 1, s_kw_protocol_cb);
    }
    Py_DECREF(tup); Py_XDECREF(cb); Py_XDECREF(proto);
    if (!data) { Py_XDECREF(oob); return false; }
    bool ok = true;
    for (Py_ssize_t i = 0; ok && i < PyList_GET_SIZE(oob); i++) ok = add_oob(PyList_GET_ITEM(oob, i));
    keep.objs.push_back(oob);   // the PickleBuffers stay alive until Finish
    if (ok) {
      Py_ssize_t n = PyBytes_GET_SIZE(data);
      if ((uint64_t)n > 0xffffffffull) {
        PyErr_SetString(PyExc_ValueError, "pickled arguments of 4 GB or more");
        ok = false;
      } else {
        void *tail = CkDynMsgReserveTail(b, (uint32_t)n);
        memcpy(tail, PyBytes_AS_STRING(data), (size_t)n);
      }
    }
    Py_DECREF(data);
    return ok;
  }
};

// Add the items of a sequence (or nothing for None) as positional arguments.
void pack_seq(Packer &pk, PyObject *args) {
  if (args == Py_None || !args) return;
  PyObject *seq = PySequence_Fast(args, "args must be a sequence");
  if (!seq) throw_py();
  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  for (Py_ssize_t i = 0; i < n; i++)
    if (!pk.add(PySequence_Fast_GET_ITEM(seq, i))) { Py_DECREF(seq); throw_py(); }
  Py_DECREF(seq);
}

// ---------------------------------------------------------------------------
// Receiver side

struct RecvCtx {
  const CkDynMsgView *v;
  bool view_mode;    // numpy views over the retained message (else copies)
  bool writable;
  PyObject *msgref = nullptr;
  RecvCtx(const CkDynMsgView *v_, bool view, bool w) : v(v_), view_mode(view && !g_copy_arrays), writable(w) {}
  ~RecvCtx() { Py_XDECREF(msgref); }
  PyObject *ref() {
    if (!msgref) {
      CkDynMsgRetain(v);
      msgref = msgref_new(v->msg);
      if (!msgref) CkDynMsgRelease(v->msg);
    }
    return msgref;
  }
};

PyObject *make_ndarray(RecvCtx &ctx, const void *data, int elem, int ndim, const npy_intp *dims, bool fortran) {
  PyArray_Descr *descr = PyArray_DescrFromType(elem_typenum(elem));
  if (!descr) return nullptr;
  if (ctx.view_mode) {
    PyObject *ref = ctx.ref();
    if (!ref) { Py_DECREF(descr); return nullptr; }
    int flags = (fortran ? NPY_ARRAY_F_CONTIGUOUS : NPY_ARRAY_C_CONTIGUOUS) | (ctx.writable ? NPY_ARRAY_WRITEABLE : 0);
    PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, dims, nullptr, (void *)data, flags, nullptr);
    if (!arr) return nullptr;
    if (PyArray_SetBaseObject((PyArrayObject *)arr, Py_NewRef(ref)) < 0) { Py_DECREF(arr); return nullptr; }
    return arr;
  }
  PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, dims, nullptr, nullptr, fortran ? NPY_ARRAY_F_CONTIGUOUS : 0, nullptr);
  if (!arr) return nullptr;
  memcpy(PyArray_DATA((PyArrayObject *)arr), data, (size_t)PyArray_NBYTES((PyArrayObject *)arr));
  return arr;
}

PyObject *remote_future(int pe, uint32_t fid) {
  if (!g_remote_future) { PyErr_SetString(PyExc_RuntimeError, "_ckdyn.init was not called"); return nullptr; }
  PyObject *a = PyLong_FromLong(pe), *b = PyLong_FromUnsignedLong(fid);
  PyObject *r = (a && b) ? call(g_remote_future, {a, b}) : nullptr;
  Py_XDECREF(a); Py_XDECREF(b);
  return r;
}

PyObject *proxy_from_slot(const CkDynMsgView *v, int i) {
  CkDynTarget t;
  CkDynViewProxy(v, i, &t);
  bool has = true;
  if (t.kind == CKDYN_ARRAY && t.u.array.ndims < 0) { t.u.array.ndims = -1 - t.u.array.ndims; has = false; }
  else if (is_group_kind(t.kind) && t.u.group.pe_or_node < 0) has = false;
  PyObject *cls = (t.chare_idx >= 0 && (size_t)t.chare_idx < g_classes.size()) ? g_classes[t.chare_idx].proxy_class : nullptr;
  if (!cls) {
    PyErr_Format(PyExc_RuntimeError, "received a proxy of chare index %d, which has no proxy class (set_proxy_class)", t.chare_idx);
    return nullptr;
  }
  return proxy_alloc(cls, t, has);
}

PyObject *slot_value(RecvCtx &ctx, int i) {
  const CkDynMsgView *v = ctx.v;
  const CkDynSlot &s = v->slots[i];
  switch (s.type) {
    case CKDYN_SLOT_NONE: Py_RETURN_NONE;
    case CKDYN_SLOT_BOOL: return PyBool_FromLong(s.v.i64 != 0);
    case CKDYN_SLOT_I64: return PyLong_FromLongLong(s.v.i64);
    case CKDYN_SLOT_U64: return PyLong_FromUnsignedLongLong(s.v.u64);
    case CKDYN_SLOT_F64: return PyFloat_FromDouble(s.v.f64);
    case CKDYN_SLOT_BYTES: {
      const char *p = (const char *)CkDynViewBytesPtr(v, i);
      if (s.tag == TAG_STR) return PyUnicode_DecodeUTF8(p, (Py_ssize_t)s.len, "strict");
      return PyBytes_FromStringAndSize(p, (Py_ssize_t)s.len);
    }
    case CKDYN_SLOT_BUFFER: {
      const int64_t *shape = nullptr;
      const void *p = CkDynViewBufferPtr(v, i, &shape);
      if ((s.tag & 0xff) == TAG_ARRAY) {
        char tc[2] = {(char)(s.tag >> 8), 0};
        PyObject *tcs = PyUnicode_FromString(tc);
        PyObject *arr = tcs ? call(g_array_type, {tcs}) : nullptr;
        Py_XDECREF(tcs);
        if (!arr) return nullptr;
        PyObject *mv = PyMemoryView_FromMemory((char *)p, (Py_ssize_t)s.len, PyBUF_READ);
        PyObject *r = mv ? PyObject_CallMethodOneArg(arr, s_frombytes, mv) : nullptr;
        Py_XDECREF(mv);
        if (!r) { Py_DECREF(arr); return nullptr; }
        Py_DECREF(r);
        return arr;
      }
      npy_intp dims[CKDYN_MAX_BUF_DIMS];
      for (int d = 0; d < s.ndim; d++) dims[d] = (npy_intp)shape[d];
      return make_ndarray(ctx, p, s.elem, s.ndim, dims, (s.bflags & CKDYN_BUF_FORTRAN) != 0);
    }
    case CKDYN_SLOT_FUTURE:
      return remote_future((int)(s.v.u64 >> 32), (uint32_t)(s.v.u64 & 0xffffffffu));
    case CKDYN_SLOT_PROXY:
      return proxy_from_slot(v, i);
    case CKDYN_SLOT_CALLBACK: {
      CallbackW w;
      CkDynViewCallback(v, i, &w.cb);
      return nb::cast(w).release().ptr();
    }
    default:
      PyErr_Format(PyExc_RuntimeError, "slot %d has a type (%d) this binding does not convert", i, (int)s.type);
      return nullptr;
  }
}

// A pickle out-of-band buffer.
PyObject *oob_buffer(RecvCtx &ctx, int i) {
  const CkDynSlot &s = ctx.v->slots[i];
  const void *p = CkDynViewBufferPtr(ctx.v, i, nullptr);
  if (ctx.view_mode) {
    npy_intp dims[1] = {(npy_intp)s.len};
    return make_ndarray(ctx, p, CKDYN_ELEM_U8, 1, dims, false);
  }
  return PyByteArray_FromStringAndSize((const char *)p, (Py_ssize_t)s.len);
}

// Value of a REDRESULT view; *empty is set for a nop reduction.
PyObject *redresult_value(RecvCtx &ctx, bool *empty) {
  const CkDynMsgView *v = ctx.v;
  *empty = false;
  if (v->nslots < 1 || v->slots[0].type != CKDYN_SLOT_BUFFER || v->slots[0].len == 0) { *empty = true; Py_RETURN_NONE; }
  const CkDynSlot &s = v->slots[0];
  const char *p = (const char *)CkDynViewBufferPtr(v, 0, nullptr);
  int elem = s.elem;
  if (elem == CKDYN_ELEM_NONE || elem == CKDYN_ELEM_BYTE) return PyBytes_FromStringAndSize(p, (Py_ssize_t)s.len);
  size_t esz = elem_size(elem);
  size_t n = s.len / esz;
  if (n == 1) {
    switch (elem) {
      case CKDYN_ELEM_I8: { int8_t x; memcpy(&x, p, 1); return PyLong_FromLong(x); }
      case CKDYN_ELEM_I16: { int16_t x; memcpy(&x, p, 2); return PyLong_FromLong(x); }
      case CKDYN_ELEM_I32: { int32_t x; memcpy(&x, p, 4); return PyLong_FromLong(x); }
      case CKDYN_ELEM_I64: { int64_t x; memcpy(&x, p, 8); return PyLong_FromLongLong(x); }
      case CKDYN_ELEM_U8: { uint8_t x; memcpy(&x, p, 1); return PyLong_FromUnsignedLong(x); }
      case CKDYN_ELEM_U16: { uint16_t x; memcpy(&x, p, 2); return PyLong_FromUnsignedLong(x); }
      case CKDYN_ELEM_U32: { uint32_t x; memcpy(&x, p, 4); return PyLong_FromUnsignedLong(x); }
      case CKDYN_ELEM_U64: { uint64_t x; memcpy(&x, p, 8); return PyLong_FromUnsignedLongLong(x); }
      case CKDYN_ELEM_F32: { float x; memcpy(&x, p, 4); return PyFloat_FromDouble(x); }
      case CKDYN_ELEM_F64: { double x; memcpy(&x, p, 8); return PyFloat_FromDouble(x); }
      case CKDYN_ELEM_C64: { float x[2]; memcpy(x, p, 8); return PyComplex_FromDoubles(x[0], x[1]); }
      case CKDYN_ELEM_C128: { double x[2]; memcpy(x, p, 16); return PyComplex_FromDoubles(x[0], x[1]); }
      case CKDYN_ELEM_BOOL: return PyBool_FromLong(p[0] != 0);
      default: {   // F16: through numpy
        RecvCtx copy(v, false, true);
        npy_intp dims[1] = {1};
        PyObject *a = make_ndarray(copy, p, elem, 1, dims, false);
        if (!a) return nullptr;
        PyObject *r = PyObject_CallMethod(a, "item", "i", 0);
        Py_DECREF(a);
        return r;
      }
    }
  }
  npy_intp dims[1] = {(npy_intp)n};
  return make_ndarray(ctx, p, elem, 1, dims, false);
}

// Positional arguments of a view (SPEC section 1, receiver side). *block
// receives the block future of slot 0 when wanted. Returns false with a
// Python error set.
bool view_args(RecvCtx &ctx, ObjVec &out, PyObject **block) {
  const CkDynMsgView *v = ctx.v;
  if (v->kind == CKDYN_MSGKIND_REDRESULT) {
    bool empty;
    PyObject *val = redresult_value(ctx, &empty);
    if (!val) return false;
    if (empty) Py_DECREF(val); else out.push(val);
    return true;
  }
  if (v->kind == CKDYN_MSGKIND_SYS || v->nslots == 0) return true;
  uint16_t flags = v->hdr ? v->hdr->flags : 0;
  int start = 0, end = v->nslots;
  const CkDynSlot *sl = v->slots;
  if ((flags & CKDYN_MSG_BLOCK) && sl[0].type == CKDYN_SLOT_FUTURE) {
    if (block) {
      *block = remote_future((int)(sl[0].v.u64 >> 32), (uint32_t)(sl[0].v.u64 & 0xffffffffu));
      if (!*block) return false;
    }
    start = 1;
  }
  if ((flags & CKDYN_MSG_CONSTRAINED) && end > start && sl[end - 1].type == CKDYN_SLOT_BYTES) end--;
  bool has_pickled = false;
  for (int i = start; i < end; i++)
    if (sl[i].type == CKDYN_SLOT_BYTES && sl[i].tag == TAG_PICKLED) { has_pickled = true; break; }
  PyObject *tail = nullptr;
  Py_ssize_t ti = 0;
  if (has_pickled) {
    if (!v->tail) { PyErr_SetString(PyExc_RuntimeError, "message with pickled arguments but no tail"); return false; }
    PyObject *bufs = PyList_New(0);
    if (!bufs) return false;
    for (int i = start; i < end; i++) {
      if (sl[i].type == CKDYN_SLOT_BUFFER && (sl[i].bflags & CKDYN_BUF_OOB)) {
        PyObject *b = oob_buffer(ctx, i);
        if (!b || PyList_Append(bufs, b) < 0) { Py_XDECREF(b); Py_DECREF(bufs); return false; }
        Py_DECREF(b);
      }
    }
    PyObject *mv = PyMemoryView_FromMemory((char *)v->tail, (Py_ssize_t)v->tail_len, PyBUF_READ);
    if (!mv) { Py_DECREF(bufs); return false; }
    PyObject *argv[2] = {mv, bufs};
    tail = PyObject_Vectorcall(g_loads, argv, 1, s_kw_buffers);
    Py_DECREF(mv); Py_DECREF(bufs);
    if (!tail) return false;
    if (!PyTuple_Check(tail)) { Py_DECREF(tail); PyErr_SetString(PyExc_RuntimeError, "pickled tail is not a tuple"); return false; }
  }
  for (int i = start; i < end; i++) {
    const CkDynSlot &s = sl[i];
    if (s.type == CKDYN_SLOT_TAIL) continue;
    if (s.type == CKDYN_SLOT_BUFFER && (s.bflags & CKDYN_BUF_OOB)) continue;
    PyObject *val;
    if (s.type == CKDYN_SLOT_BYTES && s.tag == TAG_PICKLED) {
      if (ti >= PyTuple_GET_SIZE(tail)) { Py_DECREF(tail); PyErr_SetString(PyExc_RuntimeError, "pickled tail is too short"); return false; }
      val = Py_NewRef(PyTuple_GET_ITEM(tail, ti++));
    } else {
      val = slot_value(ctx, i);
    }
    if (!val) { Py_XDECREF(tail); return false; }
    out.push(val);
  }
  Py_XDECREF(tail);
  return true;
}

// First positional argument (new reference), or None.
PyObject *first_arg(RecvCtx &ctx) {
  ObjVec a;
  if (!view_args(ctx, a, nullptr)) return nullptr;
  if (a.n == 0) Py_RETURN_NONE;
  return Py_NewRef(a.p[0]);
}

PyObject *make_header(const CkDynMsgView *v, PyObject *block) {
  PyObject *h = PyDict_New();
  if (!h) return nullptr;
  uint16_t f = v->hdr ? v->hdr->flags : 0;
  bool ok = true;
  if (block) ok &= PyDict_SetItem(h, k_block, block) == 0;
  if (f & CKDYN_MSG_BCAST) ok &= PyDict_SetItem(h, k_bcast, Py_True) == 0;
  if (f & CKDYN_MSG_BCASTRET) ok &= PyDict_SetItem(h, k_bcastret, Py_True) == 0;
  if (f & CKDYN_MSG_CREATION) ok &= PyDict_SetItem(h, k_creation, Py_True) == 0;
  if (f & CKDYN_MSG_RAY) ok &= PyDict_SetItem(h, k_is_ray, Py_True) == 0;
  if (f & PYMSG_REDRESULT) {
    PyObject *rid = PyLong_FromUnsignedLong(v->hdr->user_id);
    ok &= rid && PyDict_SetItem(h, k_custom_reducer, rid) == 0;
    Py_XDECREF(rid);
  }
  if (f & PYMSG_SINGLE) ok &= PyDict_SetItem(h, k_single, Py_True) == 0;
  if (!ok) { Py_DECREF(h); return nullptr; }
  return h;
}

// ---------------------------------------------------------------------------
// Delivery (SPEC 2.3)

// The envelope's reference number as a callback refnum: 0 when none. A
// reduction result carries its user flag, -1 by default, as the reference
// number (ckreduction.C CkSetRefNum(result, getUserFlag())), so all ones also
// means none.
// Only reduction results and system messages are read: on a CkDynMsg the
// envelope's ref field is not initialized by the runtime (CkDynMsgFinish does
// not clear it), so it holds garbage except on a foreign reduction result.
inline uint32_t callback_refnum(const CkDynMsgView *v) {
  if (v->kind == CKDYN_MSGKIND_DYN && !(v->hdr && (v->hdr->flags & PYMSG_REDRESULT))) return 0;
  return v->refnum == 0xFFFFFFFFu ? 0 : v->refnum;
}

MethodEntry &method_or_abort(int ep) {
  if (ep < 0 || (size_t)ep >= g_methods.size() || !g_methods[ep].valid) {
    PyErr_Format(PyExc_RuntimeError, "delivery to entry %d, which this binding did not register", ep);
    hook_abort("deliver");
  }
  return g_methods[ep];
}

// Flush obj's wait queues after an entry ran (auto_flush_wait_queues), as
// charm.invokeEntryMethod does on the slow path.
void flush_after(PyObject *obj, int ep) {
  PyObject *cond = g_auto_flush ? PyObject_GetAttr(obj, s_cond_next) : nullptr;
  if (!cond) {
    PyErr_Clear();
  } else {
    if (cond != Py_None) {
      PyObject *rr = PyObject_CallMethodNoArgs(obj, s_flush);
      if (!rr) {
        PyObject *exc = fetch_exc();
        PyObject *epo = PyLong_FromLong(ep);
        PyObject *r2 = call(c_process_exception, {obj, epo, exc ? exc : Py_None});
        Py_XDECREF(epo); Py_XDECREF(exc);
        if (!r2) hook_abort("charm._process_exception");
        Py_DECREF(r2);
      } else {
        Py_DECREF(rr);
      }
    }
    Py_DECREF(cond);
  }
}

void deliver_one(PyObject *obj, int ep, uint32_t ep_flags, const CkDynMsgView *v, const ObjVec &args, PyObject *block) {
  MethodEntry &m = method_or_abort(ep);
  uint16_t hflags = v->hdr ? v->hdr->flags : 0;
  Py_INCREF(obj);
  if (!g_profiling && !m.has_when && !(ep_flags & CKDYN_EP_THREADED) && hflags == 0 && callback_refnum(v) == 0 && m.func) {
    PyObject *small[9];
    PyObject **stack = args.n + 1 <= 9 ? small : (PyObject **)malloc((args.n + 1) * sizeof(PyObject *));
    stack[0] = obj;
    for (size_t i = 0; i < args.n; i++) stack[i + 1] = args.p[i];
    tls.last_ep = ep;
    PyObject *r = PyObject_Vectorcall(m.func, stack, args.n + 1, nullptr);
    if (stack != small) free(stack);
    if (!r) {
      PyObject *exc = fetch_exc();
      PyObject *epo = PyLong_FromLong(ep);
      PyObject *rr = call(c_process_exception, {obj, epo, exc ? exc : Py_None});
      Py_XDECREF(epo); Py_XDECREF(exc);
      if (!rr) hook_abort("charm._process_exception");
      Py_DECREF(rr);
    } else {
      Py_DECREF(r);
    }
    flush_after(obj, ep);
  } else if (!g_profiling && !m.has_when && (ep_flags & CKDYN_EP_THREADED) && hflags == 0 && callback_refnum(v) == 0 && m.em) {
    // coroutine with no header: em.run(obj, {}, args) (EntryMethod._run_th) directly,
    // skipping charm._invoke and invokeEntryMethod, whose other work does not apply
    if (!m.run && !(m.run = PyObject_GetAttrString(m.em, "run"))) hook_abort("deliver (EntryMethod.run)");
    PyObject *header = PyDict_New();
    PyObject *lst = header ? args.list() : nullptr;
    if (!header || !lst) hook_abort("deliver (building arguments)");
    tls.last_ep = ep;
    PyObject *r = call(m.run, {obj, header, lst});
    Py_DECREF(header); Py_DECREF(lst);
    if (!r) {
      PyObject *exc = fetch_exc();
      PyObject *epo = PyLong_FromLong(ep);
      PyObject *rr = call(c_process_exception, {obj, epo, exc ? exc : Py_None});
      Py_XDECREF(epo); Py_XDECREF(exc);
      if (!rr) hook_abort("charm._process_exception");
      Py_DECREF(rr);
    } else {
      Py_DECREF(r);
    }
    flush_after(obj, ep);
  } else {
    PyObject *header = make_header(v, block);
    PyObject *lst = header ? args.list() : nullptr;
    if (lst && callback_refnum(v) != 0) {   // a callback with a reference number: it is the first argument
      PyObject *rn = PyLong_FromUnsignedLong(callback_refnum(v));
      if (!rn || PyList_Insert(lst, 0, rn) < 0) { Py_XDECREF(rn); Py_CLEAR(lst); }
      else Py_DECREF(rn);
    }
    PyObject *epo = PyLong_FromLong(ep);
    if (!header || !lst || !epo) hook_abort("deliver (building header)");
    PyObject *r = call(c_invoke, {obj, epo, header, lst});
    Py_DECREF(header); Py_DECREF(lst); Py_DECREF(epo);
    if (!r) hook_abort("charm._invoke");
    Py_DECREF(r);
  }
  Py_DECREF(obj);
}

void h_deliver(void *, void *handle, int ep, uint32_t, uint32_t ep_flags, const CkDynMsgView *v) {
  Gil g;
  try {
    uint16_t f = v->hdr ? v->hdr->flags : 0;
    RecvCtx ctx(v, true, !(f & CKDYN_MSG_BCAST));
    ObjVec args;
    PyObject *block = nullptr;
    if (!view_args(ctx, args, &block)) hook_abort("deliver (converting arguments)");
    deliver_one((PyObject *)handle, ep, ep_flags, v, args, block);
    Py_XDECREF(block);
  } catch (...) {
    cpp_exc_to_python();
    hook_abort("deliver");
  }
}

void h_deliver_many(void *, void **handles, int n, int ep, uint32_t, uint32_t ep_flags, const CkDynMsgView *v) {
  Gil g;
  try {
    RecvCtx ctx(v, true, false);
    ObjVec args;
    PyObject *block = nullptr;
    if (!view_args(ctx, args, &block)) hook_abort("deliver_many (converting arguments)");
    for (int i = 0; i < n; i++) {
      PyObject *obj = (PyObject *)handles[i];
      if (!obj) continue;
      CkDynTarget t = target_of(obj);
      CkDynCallstackPush(&t);
      deliver_one(obj, ep, ep_flags, v, args, block);
      CkDynCallstackPop(&t);
    }
    Py_XDECREF(block);
  } catch (...) {
    cpp_exc_to_python();
    hook_abort("deliver_many");
  }
}

// ---------------------------------------------------------------------------
// Other hooks (SPEC 2.2)

void h_register_classes(void *) {
  // One PE per process until step 7 (the GIL and the runtime's per-PE threads).
  // CkDynStart does not return, so no Python exception can reach the program
  // here: abort with the message (CmiAbort prints it on stderr).
  int node_size = CkDynNodeSize(CkDynMyNode());
  if (node_size > 1) {
    char text[512];
    snprintf(text, sizeof text,
             "charm4py runs one PE per process for now, and this process has %d PEs. "
             "Use several processes with one PE each instead of +pe N in one process: "
             "lcrun -n N python app.py +pe N, or python -m charmrun.start +pN app.py.",
             node_size);
    CkDynAbort(text);   // before taking the GIL: this thread may not hold it
  }
  Gil g;
  g_in_register = true;
  PyObject *r = c_registerMainModule ? PyObject_CallNoArgs(c_registerMainModule) : nullptr;
  g_in_register = false;
  if (!r) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "charm.registerMainModule is missing");
    hook_abort("charm.registerMainModule");
  }
  Py_DECREF(r);
}

void h_mainchare(void *, void **handle_out, const CkDynTarget *t, int argc, char **argv) {
  Gil g;
  PyObject *lst = PyList_New(argc);
  for (int i = 0; lst && i < argc; i++) {
    PyObject *s = PyUnicode_DecodeFSDefault(argv[i]);
    if (!s) hook_abort("mainchare (argv)");
    PyList_SET_ITEM(lst, i, s);
  }
  PyObject *tg = make_target(*t);
  PyObject *obj = (lst && tg) ? call(c_mainchare, {tg, lst}) : nullptr;
  Py_XDECREF(lst); Py_XDECREF(tg);
  if (!obj) hook_abort("charm._mainchare");
  *handle_out = obj;
}

void *h_construct(void *, const CkDynTarget *t, const CkDynMsgView *v) {
  Gil g;
  try {
    bool p2p = (t->kind == CKDYN_CHARE);
    RecvCtx ctx(v, true, p2p);
    ObjVec args;
    PyObject *block = nullptr;
    if (!view_args(ctx, args, &block)) hook_abort("construct (converting arguments)");
    PyObject *header = make_header(v, block);
    Py_XDECREF(block);
    PyObject *lst = header ? args.list() : nullptr;
    PyObject *tg = make_target(*t);
    PyObject *epo = PyLong_FromLong(v->hdr ? v->hdr->ep : -1);
    if (!header || !lst || !tg || !epo) hook_abort("construct");
    PyObject *obj = call(c_construct, {tg, epo, header, lst});
    Py_DECREF(header); Py_DECREF(lst); Py_DECREF(tg); Py_DECREF(epo);
    if (!obj) hook_abort("charm._construct");
    return obj;
  } catch (...) {
    cpp_exc_to_python();
    hook_abort("construct");
  }
}

void h_destruct(void *, void *handle) {
  Gil g;
  PyObject *obj = (PyObject *)handle;
  auto it = tls.packed.find(handle);
  if (it != tls.packed.end()) { Py_DECREF(it->second); tls.packed.erase(it); }
  if (c_destruct) {
    PyObject *r = call(c_destruct, {obj});
    if (!r) hook_abort("charm._destruct");
    Py_DECREF(r);
  }
  Py_DECREF(obj);
}

size_t h_pack_size(void *, void *handle, int reason) {
  Gil g;
  auto it = tls.packed.find(handle);
  if ((reason & CKDYN_PUP_SIZING) || it == tls.packed.end()) {
    PyObject *ro = PyLong_FromLong(reason & 0xff);
    PyObject *data = ro ? call(c_pack, {(PyObject *)handle, ro}) : nullptr;
    Py_XDECREF(ro);
    if (!data) hook_abort("charm._pack");
    if (data == Py_None) { Py_DECREF(data); return CKDYN_PACK_REFUSED; }
    if (!PyBytes_Check(data)) { Py_DECREF(data); PyErr_SetString(PyExc_TypeError, "charm._pack must return bytes or None"); hook_abort("charm._pack"); }
    it = tls.packed.find(handle);
    if (it != tls.packed.end()) { Py_DECREF(it->second); it->second = data; }
    else tls.packed.emplace(handle, data);
    return (size_t)PyBytes_GET_SIZE(data);
  }
  return (size_t)PyBytes_GET_SIZE(it->second);
}

void h_pack(void *bd, void *handle, int reason, void *buf, size_t len) {
  if (Gil g; tls.packed.find(handle) == tls.packed.end()) {
    if (h_pack_size(bd, handle, reason) == CKDYN_PACK_REFUSED) {
      PyErr_SetString(PyExc_RuntimeError, "charm._pack refused on the packing pass");
      hook_abort("pack");
    }
  }
  Gil g;
  auto it = tls.packed.find(handle);
  PyObject *data = it->second;
  if ((size_t)PyBytes_GET_SIZE(data) != len) {
    PyErr_Format(PyExc_RuntimeError, "pack: %zu bytes kept, %zu asked", (size_t)PyBytes_GET_SIZE(data), len);
    hook_abort("pack");
  }
  memcpy(buf, PyBytes_AS_STRING(data), len);
  Py_DECREF(data);
  tls.packed.erase(it);
}

void *h_unpack(void *, const CkDynTarget *t, int reason, const void *buf, size_t len) {
  Gil g;
  PyObject *tg = make_target(*t);
  PyObject *ro = PyLong_FromLong(reason);
  PyObject *mv = PyMemoryView_FromMemory((char *)buf, (Py_ssize_t)len, PyBUF_READ);
  PyObject *obj = (tg && ro && mv) ? call(c_unpack, {tg, ro, mv}) : nullptr;
  if (mv) { PyObject *r = PyObject_CallMethod(mv, "release", nullptr); if (r) Py_DECREF(r); else PyErr_Clear(); }
  Py_XDECREF(tg); Py_XDECREF(ro); Py_XDECREF(mv);
  if (!obj) hook_abort("charm._unpack");
  return obj;
}

size_t h_pack_readonly_size(void *) {
  Gil g;
  Py_CLEAR(tls.ro_packed);
  PyObject *data = c_pack_readonlies ? PyObject_CallNoArgs(c_pack_readonlies) : PyBytes_FromStringAndSize("", 0);
  if (!data) hook_abort("charm._pack_readonlies");
  if (!PyBytes_Check(data)) { Py_DECREF(data); PyErr_SetString(PyExc_TypeError, "charm._pack_readonlies must return bytes"); hook_abort("charm._pack_readonlies"); }
  tls.ro_packed = data;
  return (size_t)PyBytes_GET_SIZE(data);
}

void h_pack_readonly(void *gl, void *buf, size_t len) {
  if (Gil g; !tls.ro_packed) h_pack_readonly_size(gl);
  Gil g;
  if ((size_t)PyBytes_GET_SIZE(tls.ro_packed) != len) {
    PyErr_SetString(PyExc_RuntimeError, "pack_readonly: length differs from the sizing pass");
    hook_abort("pack_readonly");
  }
  memcpy(buf, PyBytes_AS_STRING(tls.ro_packed), len);
  Py_CLEAR(tls.ro_packed);
}

void h_unpack_readonly(void *, const void *buf, size_t len) {
  Gil g;
  if (!c_recvReadOnly) return;
  PyObject *b = PyBytes_FromStringAndSize((const char *)buf, (Py_ssize_t)len);
  PyObject *r = b ? call(c_recvReadOnly, {b}) : nullptr;
  Py_XDECREF(b);
  if (!r) hook_abort("charm.recvReadOnly");
  Py_DECREF(r);
}

void h_reduce(void *, uint32_t rid, const CkDynMsgView **contribs, int n, CkDynMsgBuilder *result) {
  Gil g;
  try {
    PyObject *vals = PyList_New(0);
    if (!vals) hook_abort("reduce");
    for (int i = 0; i < n; i++) {
      RecvCtx ctx(contribs[i], false, true);
      PyObject *val = first_arg(ctx);
      if (!val || PyList_Append(vals, val) < 0) hook_abort("reduce (converting a contribution)");
      Py_DECREF(val);
    }
    PyObject *ro = PyLong_FromUnsignedLong(rid);
    PyObject *value = ro ? call(c_reduce_foreign, {ro, vals}) : nullptr;
    Py_XDECREF(ro); Py_DECREF(vals);
    if (!value) hook_abort("charm._reduce_foreign");
    tls.reduce_keep.release();   // the previous result was finished by CkDyn after its hook returned
    Packer pk(result);
    if (!pk.add(value) || !pk.finish()) hook_abort("reduce (packing the result)");
    Py_DECREF(value);
    // CkDyn finishes `result` after this hook returns: keep the buffer sources until the next call
    tls.reduce_keep.take(pk.keep);
    CkDynMsgSetFlags(result, PYMSG_REDRESULT);
  } catch (...) {
    cpp_exc_to_python();
    hook_abort("reduce");
  }
}

void h_future(void *, int, uint32_t id, const CkDynMsgView *v) {
  Gil g;
  try {
    PyObject *value;
    long rid = -1;
    RecvCtx ctx(v, false, true);
    if (v->kind == CKDYN_MSGKIND_REDRESULT) {
      bool empty;
      value = redresult_value(ctx, &empty);
    } else if (v->kind == CKDYN_MSGKIND_SYS) {
      value = Py_NewRef(Py_None);
    } else {
      if (v->hdr && (v->hdr->flags & PYMSG_REDRESULT)) rid = (long)v->hdr->user_id;
      value = first_arg(ctx);
    }
    if (!value) hook_abort("future (converting the value)");
    PyObject *fo = PyLong_FromUnsignedLong(id), *ro = PyLong_FromLong(rid);
    PyObject *r = (fo && ro) ? call(c_deposit_future, {fo, value, ro}) : nullptr;
    Py_XDECREF(fo); Py_XDECREF(ro); Py_DECREF(value);
    if (!r) hook_abort("charm._deposit_future");
    Py_DECREF(r);
  } catch (...) {
    cpp_exc_to_python();
    hook_abort("future");
  }
}

int h_map_proc_num(void *, void *handle, int aid, int ndims, const int *index) {
  Gil g;
  PyObject *a = PyLong_FromLong(aid);
  PyObject *idx = index_tuple(index, ndims);
  PyObject *r = (a && idx) ? call(c_map_proc_num, {(PyObject *)handle, a, idx}) : nullptr;
  Py_XDECREF(a); Py_XDECREF(idx);
  if (!r) hook_abort("charm._map_proc_num");
  long pe = PyLong_AsLong(r);
  Py_DECREF(r);
  if (pe == -1 && PyErr_Occurred()) hook_abort("charm._map_proc_num (result)");
  return (int)pe;
}

void h_resume_from_sync(void *, void *handle) {
  Gil g;
  PyObject *r = c_resume_from_sync ? call(c_resume_from_sync, {(PyObject *)handle}) : nullptr;
  if (!r) hook_abort("charm._resume_from_sync");
  Py_DECREF(r);
}

void h_error(void *, const char *what) {
  Gil g;
  PyObject *s = PyUnicode_FromString(what);
  PyObject *r = (s && c_error) ? call(c_error, {s}) : nullptr;
  Py_XDECREF(s);
  if (r) Py_DECREF(r);
  else { PyErr_Clear(); fprintf(stderr, "CkDyn error: %s\n", what); }
  flush_stdio();
}

void ccs_handler(void *, const char *name, const void *data, int len) {
  Gil g;
  PyObject *n = PyUnicode_FromString(name);
  PyObject *b = PyBytes_FromStringAndSize((const char *)data, len);
  PyObject *r = (n && b && c_callHandler) ? call(c_callHandler, {n, b}) : nullptr;
  Py_XDECREF(n); Py_XDECREF(b);
  if (!r) hook_abort("charm.callHandler");
  Py_DECREF(r);
}

void call_after_fn(void *arg, double) {
  Gil g;
  PyObject *t = PyLong_FromLong((long)(intptr_t)arg);
  PyObject *r = (t && c_triggerCallable) ? call(c_triggerCallable, {t}) : nullptr;
  Py_XDECREF(t);
  if (!r) hook_abort("charm.triggerCallable");
  Py_DECREF(r);
}

// ---------------------------------------------------------------------------
// Sends (SPEC 2.4)

void send_built(const CkDynTarget &t, bool has_elem, int ep, CkDynMsg *m) {
  if (has_elem || t.kind == CKDYN_CHARE || t.kind == CKDYN_MAINCHARE) {
    CkDynSendTarget(&t, ep, m, nullptr);
    return;
  }
  switch (t.kind) {
    case CKDYN_GROUP: case CKDYN_ARRAYMAP: CkDynSendGroup(t.u.group.gid, -1, ep, m, nullptr); break;
    case CKDYN_NODEGROUP: CkDynSendNodeGroup(t.u.group.gid, -1, ep, m, nullptr); break;
    case CKDYN_ARRAY: CkDynSendArray(t.u.array.aid, 0, nullptr, ep, m, nullptr); break;
    default: CkDynMsgFree(m); throw CkDynError("send: proxy of an unknown kind");
  }
}

bool is_bcast_proxy(const ProxyObject *p) {
  return !p->has_elem && (is_group_kind(p->t.kind) || p->t.kind == CKDYN_ARRAY);
}

// Build and send; returns false with a Python error set.
bool do_send(ProxyObject *p, int ep, PyObject *const *args, size_t nargs, uint16_t flags, PyObject *block_future,
             bool caller_flags = false) {
  if (is_bcast_proxy(p)) flags |= CKDYN_MSG_BCAST;
  else if (!caller_flags) flags &= (uint16_t)~CKDYN_MSG_BCASTRET;
  if (block_future && block_future != Py_None) flags |= CKDYN_MSG_BLOCK;
  else if (caller_flags && (flags & CKDYN_MSG_BLOCK)) {
    PyErr_SetString(PyExc_ValueError, "send: BLOCK flag without a future");
    return false;
  } else flags &= (uint16_t)~CKDYN_MSG_BLOCK;
  BuilderGuard bg(CkDynMsgBegin((int)nargs + 2, 64, 0, 0, -1));
  CkDynMsgSetFlags(bg.b, flags);
  if (flags & CKDYN_MSG_BLOCK) {
    int pe; uint32_t fid;
    if (!future_ids(block_future, &pe, &fid)) return false;
    CkDynMsgAddFuture(bg.b, pe, fid);
  }
  Packer pk(bg.b);   // declared after bg: destroyed first, after Finish or before Abandon
  for (size_t i = 0; i < nargs; i++)
    if (!pk.add(args[i])) return false;
  if (!pk.finish()) return false;
  CkDynMsg *m = bg.finish(ep);
  try {
    send_built(p->t, p->has_elem, ep, m);
  } catch (CkDynError &e) {
    PyObject *t = charm4py_error_type();
    PyErr_SetString(t, e.what());
    Py_DECREF(t);
    return false;
  }
  return true;
}

PyObject *pm_call(PMObject *pm, PyObject *proxy, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
  if (!is_proxy(proxy)) {
    PyErr_Format(PyExc_TypeError, "%U: first argument must be a proxy", pm->qualname);
    return nullptr;
  }
  bool ret = false, awaitable = false, is_ray = false;
  Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  PyObject *const *kwvals = args + nargs;
  int nuser = 0;
  for (Py_ssize_t k = 0; k < nkw; k++) {
    PyObject *name = PyTuple_GET_ITEM(kwnames, k);
    int which = 0;
    if (PyUnicode_Compare(name, s_ret) == 0) which = 1;
    else if (PyUnicode_Compare(name, s_awaitable) == 0) which = 2;
    else if (PyUnicode_Compare(name, s_is_ray) == 0) which = 3;
    if (!which) { nuser++; continue; }
    int t = PyObject_IsTrue(kwvals[k]);
    if (t < 0) return nullptr;
    if (which == 1) ret = t; else if (which == 2) awaitable = t; else is_ray = t;
  }
  std::vector<PyObject *> full;   // borrowed
  PyObject *const *pos = args;
  size_t npos = (size_t)nargs;
  if (nuser > 0 || is_ray) {
    full.assign(args, args + nargs);
    if (nuser > 0) {
      std::vector<bool> used((size_t)nkw, false);
      Py_ssize_t ndef = PyTuple_GET_SIZE(pm->defaults);
      for (int i = (int)nargs; i < pm->argcount; i++) {
        PyObject *an = PyTuple_GET_ITEM(pm->argnames, i);
        PyObject *val = nullptr;
        for (Py_ssize_t k = 0; k < nkw; k++)
          if (!used[k] && PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, k), an) == 0) { val = kwvals[k]; used[k] = true; break; }
        if (!val) {
          Py_ssize_t di = i - pm->argcount + ndef;
          if (di < 0) {
            PyErr_Format(PyExc_TypeError, "%U: missing value for parameter '%U'", pm->qualname, an);
            return nullptr;
          }
          val = PyTuple_GET_ITEM(pm->defaults, di);
        }
        full.push_back(val);
      }
      for (Py_ssize_t k = 0; k < nkw; k++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, k);
        if (used[k] || PyUnicode_Compare(name, s_ret) == 0 || PyUnicode_Compare(name, s_awaitable) == 0 ||
            PyUnicode_Compare(name, s_is_ray) == 0)
          continue;
        PyErr_Format(PyExc_TypeError, "%U got an unexpected keyword argument '%U'", pm->qualname, name);
        return nullptr;
      }
    }
  }
  PyObject *future = nullptr, *ray_future = nullptr;
  if (ret || awaitable) {
    future = PyObject_CallMethodNoArgs(g_charm, s_Future);
    if (!future) return nullptr;
  }
  if (is_ray) {   // charm.createFuture(store=True), appended as the last argument
    PyObject *argv[1] = {Py_True};
    PyObject *meth = PyObject_GetAttr(g_charm, s_createFuture);
    ray_future = meth ? PyObject_Vectorcall(meth, argv, 0, s_kw_store) : nullptr;
    Py_XDECREF(meth);
    if (!ray_future) { Py_XDECREF(future); return nullptr; }
    full.push_back(ray_future);
  }
  if (!full.empty() || nuser > 0 || is_ray) { pos = full.data(); npos = full.size(); }
  ProxyObject *p = (ProxyObject *)proxy;
  uint16_t flags = 0;
  if (is_ray) flags |= CKDYN_MSG_RAY;
  if (ret && is_bcast_proxy(p)) flags |= CKDYN_MSG_BCASTRET;
  bool ok = do_send(p, pm->ep, pos, npos, flags, future);
  if (!ok) { Py_XDECREF(future); Py_XDECREF(ray_future); return nullptr; }
  if (ray_future) { Py_XDECREF(future); return ray_future; }
  if (future) return future;
  Py_RETURN_NONE;
}

PyObject *pm_vectorcall(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  PMObject *pm = (PMObject *)self;
  if (nargs < 1) {
    PyErr_Format(PyExc_TypeError, "%U needs a proxy as its first argument", pm->qualname);
    return nullptr;
  }
  return pm_call(pm, args[0], args + 1, nargs - 1, kwnames);
}

PyObject *bpm_vectorcall(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
  BPMObject *b = (BPMObject *)self;
  return pm_call(b->pm, b->proxy, args, PyVectorcall_NARGS(nargsf), kwnames);
}

PyObject *bpm_new(PyObject *proxy, PMObject *pm) {
  BPMObject *b = PyObject_New(BPMObject, &BPMType);
  if (!b) return nullptr;
  b->vectorcall = bpm_vectorcall;
  b->proxy = Py_NewRef(proxy);
  b->pm = (PMObject *)Py_NewRef((PyObject *)pm);
  return (PyObject *)b;
}

PyObject *pm_descr_get(PyObject *self, PyObject *obj, PyObject *) {
  if (!obj || obj == Py_None) return Py_NewRef(self);
  return bpm_new(obj, (PMObject *)self);
}

void pm_dealloc(PyObject *self) {
  PMObject *pm = (PMObject *)self;
  Py_XDECREF(pm->argnames); Py_XDECREF(pm->defaults); Py_XDECREF(pm->name); Py_XDECREF(pm->qualname);
  Py_TYPE(self)->tp_free(self);
}

PyObject *pm_get_ep(PyObject *self, void *) { return PyLong_FromLong(((PMObject *)self)->ep); }
PyObject *pm_get_name(PyObject *self, void *) { return Py_NewRef(((PMObject *)self)->name); }
PyObject *pm_get_qualname(PyObject *self, void *) { return Py_NewRef(((PMObject *)self)->qualname); }
PyObject *pm_repr(PyObject *self) { return PyUnicode_FromFormat("<proxy method %U ep=%d>", ((PMObject *)self)->qualname, ((PMObject *)self)->ep); }

PyGetSetDef pm_getset[] = {
    {"ep", pm_get_ep, nullptr, nullptr, nullptr},
    {"__name__", pm_get_name, nullptr, nullptr, nullptr},
    {"__qualname__", pm_get_qualname, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

void bpm_dealloc(PyObject *self) {
  BPMObject *b = (BPMObject *)self;
  Py_XDECREF(b->proxy); Py_XDECREF((PyObject *)b->pm);
  Py_TYPE(self)->tp_free(self);
}
PyObject *bpm_get_ep(PyObject *self, void *) { return PyLong_FromLong(((BPMObject *)self)->pm->ep); }
PyObject *bpm_get_self(PyObject *self, void *) { return Py_NewRef(((BPMObject *)self)->proxy); }
PyObject *bpm_get_func(PyObject *self, void *) { return Py_NewRef((PyObject *)((BPMObject *)self)->pm); }
PyObject *bpm_get_name(PyObject *self, void *) { return Py_NewRef(((BPMObject *)self)->pm->name); }
PyObject *bpm_get_qualname(PyObject *self, void *) { return Py_NewRef(((BPMObject *)self)->pm->qualname); }

PyObject *bpm_richcompare(PyObject *a, PyObject *b, int op) {
  if ((op != Py_EQ && op != Py_NE) || !PyObject_TypeCheck(b, &BPMType)) Py_RETURN_NOTIMPLEMENTED;
  BPMObject *x = (BPMObject *)a, *y = (BPMObject *)b;
  int eq = 0;
  if (x->pm->ep == y->pm->ep) {
    eq = PyObject_RichCompareBool(x->proxy, y->proxy, Py_EQ);
    if (eq < 0) return nullptr;
  }
  if ((eq != 0) == (op == Py_EQ)) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

Py_hash_t bpm_hash(PyObject *self) {
  BPMObject *b = (BPMObject *)self;
  Py_hash_t h = PyObject_Hash(b->proxy);
  if (h == -1) return -1;
  h = h * 1000003 ^ (Py_hash_t)b->pm->ep;
  return h == -1 ? -2 : h;
}

PyObject *g_rebuild_bound = nullptr;

// __reduce__: (_ckdyn._rebuild_bound, (proxy, ep))
PyObject *bpm_reduce(PyObject *self, PyObject *) {
  BPMObject *b = (BPMObject *)self;
  if (!g_rebuild_bound) { PyErr_SetString(PyExc_RuntimeError, "_ckdyn._rebuild_bound missing"); return nullptr; }
  return Py_BuildValue("(O(Oi))", g_rebuild_bound, b->proxy, b->pm->ep);
}

PyObject *rebuild_bound(PyObject *, PyObject *const *a, Py_ssize_t nargs) {
  if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "_rebuild_bound(proxy, ep)"); return nullptr; }
  long ep = PyLong_AsLong(a[1]);
  if (ep == -1 && PyErr_Occurred()) return nullptr;
  auto it = g_pm_name.find((int)ep);
  if (it != g_pm_name.end()) {
    PyObject *r = PyObject_GetAttr(a[0], it->second);
    if (r && PyObject_TypeCheck(r, &BPMType) && ((BPMObject *)r)->pm->ep == ep) return r;
    Py_XDECREF(r);
    PyErr_Clear();
  }
  // scan the class dicts along the MRO
  PyObject *mro = Py_TYPE(a[0])->tp_mro;
  for (Py_ssize_t i = 0; mro && i < PyTuple_GET_SIZE(mro); i++) {
    PyObject *d = ((PyTypeObject *)PyTuple_GET_ITEM(mro, i))->tp_dict;
    PyObject *k, *v;
    Py_ssize_t pos = 0;
    while (d && PyDict_Next(d, &pos, &k, &v))
      if (PyObject_TypeCheck(v, &PMType) && ((PMObject *)v)->ep == ep) return bpm_new(a[0], (PMObject *)v);
  }
  PyErr_Format(PyExc_AttributeError, "%R has no proxy method with entry index %ld", (PyObject *)Py_TYPE(a[0]), ep);
  return nullptr;
}

PyObject *bpm_repr(PyObject *self) {
  BPMObject *b = (BPMObject *)self;
  return PyUnicode_FromFormat("<bound proxy method %U of %R>", b->pm->qualname, b->proxy);
}

PyGetSetDef bpm_getset[] = {
    {"ep", bpm_get_ep, nullptr, nullptr, nullptr},
    {"__self__", bpm_get_self, nullptr, nullptr, nullptr},
    {"__func__", bpm_get_func, nullptr, nullptr, nullptr},
    {"__name__", bpm_get_name, nullptr, nullptr, nullptr},
    {"__qualname__", bpm_get_qualname, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyMethodDef bpm_methods[] = {
    {"__reduce__", (PyCFunction)bpm_reduce, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

// ---------------------------------------------------------------------------
// Callbacks from Python objects

CkDynCallback callback_from(nb::handle h) {
  CkDynCallback cb;
  memset(&cb, 0, sizeof cb);
  PyObject *o = h.ptr();
  if (o == Py_None) { cb.kind = CKDYN_CB_IGNORE; return cb; }
  if (PyObject_TypeCheck(o, &BPMType)) {
    BPMObject *b = (BPMObject *)o;
    ProxyObject *p = (ProxyObject *)b->proxy;
    cb.kind = CKDYN_CB_TARGET;
    cb.t = p->t;
    cb.ep = b->pm->ep;
    cb.bcast = is_bcast_proxy(p) ? 1 : 0;
    return cb;
  }
  CallbackW *w = nullptr;
  if (nb::try_cast<CallbackW *>(h, w, false) && w) return w->cb;
  if (g_CollFuture_type) {
    int r = PyObject_IsInstance(o, g_CollFuture_type);
    if (r < 0) throw_py();
    if (r) {   // the proxy's _coll_future_deposit_result with refnum = fid
      nb::object proxy = nb::borrow(o).attr("proxy");
      nb::object bpm = proxy.attr("_coll_future_deposit_result");
      int pe; uint32_t fid;
      if (!future_ids(o, &pe, &fid)) throw_py();
      if (!PyObject_TypeCheck(bpm.ptr(), &BPMType)) throw nb::type_error("CollectiveFuture: proxy._coll_future_deposit_result is not a proxy method");
      CkDynCallback c2 = callback_from(bpm);
      c2.refnum = fid;
      return c2;
    }
  }
  if (g_Future_type) {
    int r = PyObject_IsInstance(o, g_Future_type);
    if (r < 0) throw_py();
    if (r) {
      int pe; uint32_t fid;
      if (!future_ids(o, &pe, &fid)) throw_py();
      cb.kind = CKDYN_CB_FUTURE; cb.pe = pe; cb.id = fid;
      return cb;
    }
  }
  throw nb::type_error("expected a bound proxy method, a Future, a _ckdyn.Callback or None");
}

// ---------------------------------------------------------------------------
// Worker pool (SPEC 2.6)

// run function of every worker greenlet: worker_main(fn, args)
PyObject *worker_main(PyObject *, PyObject *const *a, Py_ssize_t nargs) {
  if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "worker_main(fn, args)"); return nullptr; }
  PyObject *fn = Py_NewRef(a[0]), *args = Py_NewRef(a[1]);
  for (;;) {
    PyObject *r = PyObject_Call(fn, args, nullptr);
    Py_DECREF(fn); Py_DECREF(args);
    PyObject *result;
    if (r) {
      Py_DECREF(r);
      result = Py_NewRef(g_DONE);
    } else {
      PyObject *exc = fetch_exc();
      if (exc && PyErr_GivenExceptionMatches(exc, PyExc_GreenletExit)) {   // being killed
        restore_exc(exc);
        return nullptr;
      }
      result = PyTuple_Pack(2, g_EXC, exc ? exc : Py_None);
      Py_XDECREF(exc);
      if (!result) return nullptr;
    }
    PyGreenlet *self = PyGreenlet_GetCurrent();
    tls.idle.push_back((PyObject *)self);
    PyGreenlet *parent = PyGreenlet_GetParent(self);
    if (!parent) { Py_DECREF(result); if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "worker without a parent"); return nullptr; }
    PyObject *argt = PyTuple_Pack(1, result);
    Py_DECREF(result);
    PyObject *next = argt ? PyGreenlet_Switch(parent, argt, nullptr) : nullptr;
    Py_XDECREF(argt);
    Py_DECREF(parent);
    if (!next) {
      if (PyErr_ExceptionMatches(PyExc_GreenletExit)) { PyErr_Clear(); Py_RETURN_NONE; }
      return nullptr;
    }
    if (!PyTuple_Check(next) || PyTuple_GET_SIZE(next) != 2) {
      Py_DECREF(next);
      PyErr_SetString(PyExc_RuntimeError, "idle worker greenlet switched to without a task");
      return nullptr;
    }
    fn = Py_NewRef(PyTuple_GET_ITEM(next, 0));
    args = Py_NewRef(PyTuple_GET_ITEM(next, 1));
    Py_DECREF(next);
  }
}

PyMethodDef worker_main_def = {"_worker_main", (PyCFunction)(void (*)(void))worker_main, METH_FASTCALL, nullptr};

// True on DONE, False when paused; re-raises a task's exception.
PyObject *switch_result(PyObject *r) {
  if (!r) return nullptr;
  if (r == g_DONE) { Py_DECREF(r); Py_RETURN_TRUE; }
  if (PyTuple_CheckExact(r) && PyTuple_GET_SIZE(r) == 2 && PyTuple_GET_ITEM(r, 0) == g_EXC) {
    PyObject *exc = Py_NewRef(PyTuple_GET_ITEM(r, 1));
    Py_DECREF(r);
    restore_exc(exc);
    return nullptr;
  }
  Py_DECREF(r);
  Py_RETURN_FALSE;
}

// Switch into worker w with argument tuple argt, bracketed by
// CkDynCoroResume/CkDynCoroSuspend for the task's chare (rec): Resume right
// before the switch, Suspend right after it returns to this frame (the task
// paused by switching to its parent, or finished). A finished task's record is
// erased. Returns switch_result's value.
PyObject *switch_into(PyGreenlet *w, PyObject *argt, const PeState::CoroRec *rec) {
  PyObject *r;
  if (rec) {
    PeState::CoroRec c = *rec;   // the map may rehash while the task runs
    CkDynCoroResume(&c.t, c.ep, c.id);
    r = PyGreenlet_Switch(w, argt, nullptr);
    CkDynCoroSuspend(&c.t);
  } else {
    r = PyGreenlet_Switch(w, argt, nullptr);
  }
  PyObject *res = switch_result(r);
  if (res != Py_False) tls.coros.erase((PyObject *)w);   // finished (True) or raised (NULL)
  return res;
}

// spawn(ep, fn, obj, *args): run fn(obj, *args) on a pooled worker greenlet.
// ep is the entry index the task runs and obj the chare (its _ckdyn_target
// names it to CkDynCoroResume).
PyObject *py_spawn(PyObject *, PyObject *const *a, Py_ssize_t nargs) {
  if (nargs < 3) { PyErr_SetString(PyExc_TypeError, "spawn(ep, fn, obj, *args)"); return nullptr; }
  int ep = (int)PyLong_AsLong(a[0]);
  if (ep == -1 && PyErr_Occurred()) return nullptr;
  PeState::CoroRec rec;
  try {
    rec.t = target_of(a[2]);
  } catch (nb::python_error &e) {
    e.restore();
    return nullptr;
  } catch (...) {
    cpp_exc_to_python();
    return nullptr;
  }
  rec.ep = ep;
  rec.id = tls.next_coro_id++;
  if (!tls.main_gr) tls.main_gr = (PyObject *)PyGreenlet_GetCurrent();
  PyObject *args = PyTuple_New(nargs - 2);
  if (!args) return nullptr;
  for (Py_ssize_t i = 2; i < nargs; i++) PyTuple_SET_ITEM(args, i - 2, Py_NewRef(a[i]));
  PyGreenlet *cur = PyGreenlet_GetCurrent();
  PyGreenlet *w;
  if (!tls.idle.empty()) {
    w = (PyGreenlet *)tls.idle.back();
    tls.idle.pop_back();
    if (PyGreenlet_SetParent(w, cur) < 0) { Py_DECREF(w); Py_DECREF(cur); Py_DECREF(args); return nullptr; }
  } else {
    w = PyGreenlet_New(g_worker_main, cur);
    if (!w) { Py_DECREF(cur); Py_DECREF(args); return nullptr; }
  }
  Py_DECREF(cur);
  PyObject *task = PyTuple_Pack(2, a[1], args);
  Py_DECREF(args);
  PyObject *r = nullptr;
  if (task) {
    tls.coros[(PyObject *)w] = rec;
    r = switch_into(w, task, &rec);
  }
  Py_XDECREF(task);
  Py_DECREF(w);
  return r;
}

// resume(gr, value): continue a paused task with value; uses the chare and
// entry recorded at spawn.
PyObject *py_resume(PyObject *, PyObject *const *a, Py_ssize_t nargs) {
  if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "resume(gr, value)"); return nullptr; }
  if (!PyGreenlet_Check(a[0])) { PyErr_SetString(PyExc_TypeError, "resume: not a greenlet"); return nullptr; }
  PyObject *argt = PyTuple_Pack(1, a[1]);
  if (!argt) return nullptr;
  auto it = tls.coros.find(a[0]);
  PyGreenlet *w = (PyGreenlet *)Py_NewRef(a[0]);   // a[0] may lose its last other reference while it runs
  PyObject *r = switch_into(w, argt, it == tls.coros.end() ? nullptr : &it->second);
  Py_DECREF(w);
  Py_DECREF(argt);
  return r;
}

PyMethodDef raw_functions[] = {
    {"spawn", (PyCFunction)(void (*)(void))py_spawn, METH_FASTCALL, "spawn(ep, fn, obj, *args) -> bool"},
    {"resume", (PyCFunction)(void (*)(void))py_resume, METH_FASTCALL, "resume(gr, value) -> bool"},
    {"_rebuild_bound", (PyCFunction)(void (*)(void))rebuild_bound, METH_FASTCALL, "unpickle a BoundProxyMethod"},
    {nullptr, nullptr, 0, nullptr}};

// ---------------------------------------------------------------------------
// Module functions (SPEC 2.7)

PyObject *getattr_or_null(PyObject *o, const char *name) {
  PyObject *r = PyObject_GetAttrString(o, name);
  if (!r) PyErr_Clear();
  return r;
}

void set_hooks() {
  memset(&g_hooks, 0, sizeof g_hooks);
  g_hooks.version = CKDYN_API_VERSION;
  g_hooks.size = (uint32_t)sizeof(CkDynHooks);
  g_hooks.global = nullptr;
  g_hooks.register_classes = h_register_classes;
  g_hooks.construct = h_construct;
  g_hooks.destruct = h_destruct;
  g_hooks.deliver = h_deliver;
  g_hooks.deliver_many = h_deliver_many;
  g_hooks.mainchare = h_mainchare;
  g_hooks.pack_size = h_pack_size;
  g_hooks.pack = h_pack;
  g_hooks.unpack = h_unpack;
  g_hooks.pack_readonly_size = h_pack_readonly_size;
  g_hooks.pack_readonly = h_pack_readonly;
  g_hooks.unpack_readonly = h_unpack_readonly;
  g_hooks.reduce = h_reduce;
  g_hooks.future = h_future;
  g_hooks.section_joined = nullptr;
  g_hooks.map_proc_num = h_map_proc_num;
  g_hooks.resume_from_sync = h_resume_from_sync;
  g_hooks.error = h_error;
}

void mod_init(nb::handle charm, nb::handle options, nb::handle future_type, nb::handle remote_fut, nb::handle coll_type) {
  if (CkDynApiVersion() != CKDYN_API_VERSION)
    throw CkDynError("libcharm has CkDyn API version " + std::to_string(CkDynApiVersion()) + ", this binding needs " +
                     std::to_string(CKDYN_API_VERSION));
  if (CkDynRefnumBytes() != 4)
    throw CkDynError("libcharm was built with a " + std::to_string(CkDynRefnumBytes()) +
                     "-byte reference number; charm4py needs REFNUM_TYPE=\"unsigned int\" (4 bytes)");
  Py_XSETREF(g_charm, Py_NewRef(charm.ptr()));
  Py_XSETREF(g_options, Py_NewRef(options.ptr()));
  PyObject *c = charm.ptr();
#define LOOKUP(var, name) Py_XSETREF(var, getattr_or_null(c, name))
  LOOKUP(c_registerMainModule, "registerMainModule");
  LOOKUP(c_construct, "_construct");
  LOOKUP(c_mainchare, "_mainchare");
  LOOKUP(c_invoke, "_invoke");
  LOOKUP(c_process_exception, "_process_exception");
  LOOKUP(c_deposit_future, "_deposit_future");
  LOOKUP(c_reduce_foreign, "_reduce_foreign");
  LOOKUP(c_pack, "_pack");
  LOOKUP(c_unpack, "_unpack");
  LOOKUP(c_pack_readonlies, "_pack_readonlies");
  LOOKUP(c_recvReadOnly, "recvReadOnly");
  LOOKUP(c_map_proc_num, "_map_proc_num");
  LOOKUP(c_resume_from_sync, "_resume_from_sync");
  LOOKUP(c_destruct, "_destruct");
  LOOKUP(c_triggerCallable, "triggerCallable");
  LOOKUP(c_callHandler, "callHandler");
  LOOKUP(c_error, "_error");
#undef LOOKUP
  if (future_type.is_none() || remote_fut.is_none()) {
    nb::module_ th = nb::module_::import_("charm4py.threads");
    if (future_type.is_none()) future_type = th.attr("Future");
    if (remote_fut.is_none()) remote_fut = th.attr("remote_future");
    if (coll_type.is_none() && nb::hasattr(th, "CollectiveFuture")) coll_type = th.attr("CollectiveFuture");
    Py_XSETREF(g_Future_type, Py_NewRef(future_type.ptr()));
    Py_XSETREF(g_remote_future, Py_NewRef(remote_fut.ptr()));
  } else {
    Py_XSETREF(g_Future_type, Py_NewRef(future_type.ptr()));
    Py_XSETREF(g_remote_future, Py_NewRef(remote_fut.ptr()));
  }
  if (!coll_type.is_none()) Py_XSETREF(g_CollFuture_type, Py_NewRef(coll_type.ptr()));
  static bool hooks_installed = false;
  if (!hooks_installed) {
    set_hooks();
    CkDynInit(&g_hooks);
    hooks_installed = true;
  }
}

bool option_true(const char *name) {
  if (!g_options) return false;
  PyObject *v = getattr_or_null(g_options, name);
  if (!v) return false;
  int t = PyObject_IsTrue(v);
  Py_DECREF(v);
  if (t < 0) { PyErr_Clear(); return false; }
  return t != 0;
}

void mod_start(nb::list argv) {
  if (!g_charm) throw CkDynError("_ckdyn.start before _ckdyn.init");
  if (option_true("remote_exec")) CkDynAllowLateRegistration();
  g_profiling = option_true("profiling");
  g_copy_arrays = option_true("copy_received_arrays");
  g_auto_flush = option_true("auto_flush_wait_queues");
  if (!tls.main_gr) tls.main_gr = (PyObject *)PyGreenlet_GetCurrent();
  size_t n = nb::len(argv);
  char **cargv = (char **)calloc(n + 1, sizeof(char *));
  for (size_t i = 0; i < n; i++) cargv[i] = strdup(nb::cast<std::string>(nb::str(argv[i])).c_str());
  atexit(flush_at_exit);
  g_started = true;
  flush_stdio();
  PyThreadState *ts = PyEval_SaveThread();
  CkDynStart((int)n, cargv);   // does not return
  PyEval_RestoreThread(ts);
}

nb::tuple mod_register_class(nb::handle cls, int kind, int ndims, nb::list entries) {
  size_t n = nb::len(entries);
  if (n < 1) throw nb::value_error("register_class: entries[0] must be the constructor");
  std::vector<nb::tuple> ents;
  for (size_t i = 0; i < n; i++) {
    nb::tuple e = nb::cast<nb::tuple>(entries[i]);
    if (nb::len(e) != 8) throw nb::value_error("register_class: each entry is (func, em, name, flags, has_when, argcount, argnames, defaults)");
    ents.push_back(e);
  }
  CkDynEntryDesc *descs = new CkDynEntryDesc[n];   // kept for the run
  for (size_t i = 0; i < n; i++) {
    descs[i].name = strdup(nb::cast<std::string>(nb::str(ents[i][2])).c_str());
    descs[i].flags = nb::cast<uint32_t>(ents[i][3]);
    descs[i].user_id = (uint32_t)i;
  }
  CkDynClassDesc d;
  memset(&d, 0, sizeof d);
  d.name = strdup(nb::cast<std::string>(nb::str(cls.attr("__name__"))).c_str());
  d.kind = (CkDynKind)kind;
  d.ndims = ndims;
  d.num_entries = (int)n;
  d.entries = descs;
  d.binding_data = nullptr;
  d.hooks = nullptr;
  int first_ep = -1, idx;
  if (g_in_register) idx = CkDynRegisterClass(&d, &first_ep);
  else if (g_started) idx = CkDynRegisterClassLate(&d, &first_ep);
  else throw CkDynError("register_class: called before start (classes are registered from charm.registerMainModule)");
  if ((size_t)(first_ep + n) > g_methods.size()) g_methods.resize(first_ep + n);
  for (size_t i = 0; i < n; i++) {
    MethodEntry &m = g_methods[first_ep + i];
    nb::tuple &e = ents[i];
    m.func = e[0].is_none() ? nullptr : Py_NewRef(e[0].ptr());
    m.em = Py_NewRef(e[1].ptr());
    m.name = Py_NewRef(e[2].ptr());
    m.flags = descs[i].flags;
    m.has_when = nb::cast<bool>(e[4]);
    m.argcount = nb::cast<int>(e[5]);
    m.argnames = PySequence_Tuple(e[6].ptr());
    m.defaults = e[7].is_none() ? PyTuple_New(0) : PySequence_Tuple(e[7].ptr());
    if (!m.argnames || !m.defaults) throw_py();
    m.valid = true;
  }
  if ((size_t)idx >= g_classes.size()) g_classes.resize(idx + 1);
  ClassEntry &ce = g_classes[idx];
  ce.cls = Py_NewRef(cls.ptr());
  ce.kind = kind;
  ce.first_ep = first_ep;
  ce.nentries = (int)n;
  return nb::make_tuple(idx, first_ep);
}

ClassEntry &class_or_throw(int idx) {
  if (idx < 0 || (size_t)idx >= g_classes.size() || !g_classes[idx].cls)
    throw CkDynError("unknown chare index " + std::to_string(idx));
  return g_classes[idx];
}

void mod_set_proxy_class(int idx, nb::handle cls) {
  ClassEntry &ce = class_or_throw(idx);
  if (!PyType_Check(cls.ptr()) || !PyType_IsSubtype((PyTypeObject *)cls.ptr(), &ProxyType))
    throw nb::type_error("set_proxy_class: not a _ckdyn.Proxy subclass");
  Py_XSETREF(ce.proxy_class, Py_NewRef(cls.ptr()));
  g_proxy_type_idx[(PyTypeObject *)cls.ptr()] = idx;
}

nb::object mod_make_proxy_method(int ep, int argcount, nb::handle argnames, nb::handle defaults, nb::handle name, nb::handle qualname) {
  PMObject *pm = PyObject_New(PMObject, &PMType);
  if (!pm) throw_py();
  pm->vectorcall = pm_vectorcall;
  pm->ep = ep;
  pm->argcount = argcount;
  pm->argnames = PySequence_Tuple(argnames.ptr());
  pm->defaults = defaults.is_none() ? PyTuple_New(0) : PySequence_Tuple(defaults.ptr());
  pm->name = Py_NewRef(name.ptr());
  pm->qualname = Py_NewRef(qualname.ptr());
  nb::object r = nb::steal((PyObject *)pm);
  if (!pm->argnames || !pm->defaults) throw_py();
  auto it = g_pm_name.find(ep);
  if (it != g_pm_name.end()) Py_DECREF(it->second);
  g_pm_name[ep] = Py_NewRef(name.ptr());
  return r;
}

// A constructor or creation message.
CkDynMsg *build_msg(int ep, uint16_t flags, nb::handle future, nb::handle args, int group_dep,
                    const std::vector<int32_t> *pes) {
  bool has_future = future.is_valid() && !future.is_none();
  if (has_future) flags |= CKDYN_MSG_BLOCK;
  if (flags & CKDYN_MSG_CREATION && !has_future) throw CkDynError("CREATION needs a future");
  if (pes) flags |= CKDYN_MSG_CONSTRAINED;
  BuilderGuard bg(CkDynMsgBegin(8, 64, 0, 0, group_dep));
  CkDynMsgSetFlags(bg.b, flags);
  if (has_future) {
    int pe; uint32_t fid;
    if (!future_ids(future.ptr(), &pe, &fid)) throw_py();
    CkDynMsgAddFuture(bg.b, pe, fid);
  }
  Packer pk(bg.b);
  pack_seq(pk, args.ptr());
  if (!pk.finish()) throw_py();
  if (pes) CkDynMsgAddBytes(bg.b, pes->data(), (uint32_t)(pes->size() * sizeof(int32_t)), 0);
  return bg.finish(ep);
}

std::vector<int> int_list(nb::handle seq) {
  std::vector<int> v;
  for (nb::handle h : seq) v.push_back(nb::cast<int>(h));
  return v;
}

PyObject *mod_send(nb::handle proxy, int ep, nb::handle args, int flags, nb::handle future) {
  if (!is_proxy(proxy.ptr())) throw nb::type_error("send: not a proxy");
  PyObject *seq = args.is_none() ? PyTuple_New(0) : PySequence_Fast(args.ptr(), "send: args must be a sequence");
  if (!seq) throw_py();
  bool ok = do_send((ProxyObject *)proxy.ptr(), ep, PySequence_Fast_ITEMS(seq), (size_t)PySequence_Fast_GET_SIZE(seq),
                    (uint16_t)flags, future.is_none() ? nullptr : future.ptr(), true);
  Py_DECREF(seq);
  if (!ok) throw_py();
  Py_RETURN_NONE;
}

nb::object mod_create_chare(int idx, int on_pe, nb::handle args, int flags, nb::handle future) {
  ClassEntry &ce = class_or_throw(idx);
  CkDynMsg *m = build_msg(ce.first_ep, (uint16_t)flags, future, args, -1, nullptr);
  CkDynTarget out;
  memset(&out, 0, sizeof out);
  CkDynCreateChare(idx, on_pe, m, &out);
  return nb::steal(make_target(out));
}

int mod_create_group(int idx, nb::handle args, int flags, nb::handle future, nb::handle pes) {
  ClassEntry &ce = class_or_throw(idx);
  std::vector<int32_t> pv;
  if (!pes.is_none()) for (int p : int_list(pes)) pv.push_back(p);
  CkDynMsg *m = build_msg(ce.first_ep, (uint16_t)flags, future, args, -1, pes.is_none() ? nullptr : &pv);
  return CkDynCreateGroup(idx, m);
}

int mod_create_nodegroup(int idx, nb::handle args, int flags, nb::handle future) {
  ClassEntry &ce = class_or_throw(idx);
  CkDynMsg *m = build_msg(ce.first_ep, (uint16_t)flags, future, args, -1, nullptr);
  return CkDynCreateNodeGroup(idx, m);
}

int mod_create_array(int idx, nb::handle dims, int ndims, nb::handle args, int map_gid, bool use_at_sync, int flags, nb::handle future) {
  ClassEntry &ce = class_or_throw(idx);
  std::vector<int> dv;
  if (!dims.is_none()) dv = int_list(dims);
  if (use_at_sync) flags |= CKDYN_MSG_USESATSYNC;
  CkDynMsg *m = build_msg(ce.first_ep, (uint16_t)flags, future, args, map_gid >= 0 ? map_gid : -1, nullptr);
  CkDynArrayOpts o;
  memset(&o, 0, sizeof o);
  o.ndims = ndims;
  o.dims = dims.is_none() ? nullptr : dv.data();
  o.map_gid = map_gid >= 0 ? map_gid : -1;
  o.bind_to_aid = -1;
  int aid = CkDynCreateArray(idx, m, &o);
  g_array_class[aid] = idx;
  return aid;
}

void mod_array_insert(int aid, nb::handle index, int on_pe, nb::handle args, int flags, nb::handle future, bool use_at_sync, int chare_idx) {
  if (chare_idx < 0) {
    auto it = g_array_class.find(aid);
    if (it == g_array_class.end())
      throw CkDynError("array_insert: the class of array " + std::to_string(aid) +
                       " is not known in this process; pass chare_idx");
    chare_idx = it->second;
  }
  ClassEntry &ce = class_or_throw(chare_idx);
  std::vector<int> iv = int_list(index);
  if (iv.empty() || iv.size() > CKDYN_MAX_DIMS) throw nb::value_error("array index must have 1 to 6 dimensions");
  if (use_at_sync) flags |= CKDYN_MSG_USESATSYNC;
  flags &= ~CKDYN_MSG_CREATION;
  CkDynMsg *m = build_msg(ce.first_ep, (uint16_t)flags, future, args, -1, nullptr);
  CkDynArrayInsert(aid, (int)iv.size(), iv.data(), on_pe, m);
}

void mod_destroy_element(int aid, nb::handle index) {
  std::vector<int> iv = int_list(index);
  CkDynArrayDestroyElement(aid, (int)iv.size(), iv.data());
}

void mod_contribute(nb::handle obj, nb::handle data, int rtype, nb::handle target) {
  CkDynTarget me = target_of(obj.ptr());
  CkDynCallback cb = callback_from(target);
  CkDynRedData d;
  memset(&d, 0, sizeof d);
  d.type = rtype;
  if (rtype == CkDynForeignReducerType()) {
    nb::tuple t = nb::cast<nb::tuple>(data);
    if (nb::len(t) != 2) throw nb::value_error("foreign contribution must be (reducer_id, value)");
    uint32_t rid = nb::cast<uint32_t>(t[0]);
    BuilderGuard bg(CkDynMsgBegin(2, 64, 0, 0, -1));
    CkDynMsgSetUserId(bg.b, rid);
    // marked as a result already: with one contributor Charm++ never calls the reducer
    CkDynMsgSetFlags(bg.b, PYMSG_REDRESULT);
    Packer pk(bg.b);
    if (!pk.add(t[1].ptr()) || !pk.finish()) throw_py();
    CkDynMsg *m = bg.finish(0);
    uint32_t len = 0;
    d.data = CkDynMsgPayload(m, &len);
    d.len = len;
    CkDynContribute(&me, &d, &cb);
    CkDynMsgFree(m);
    return;
  }
  if (data.is_none()) {
    CkDynContribute(&me, &d, &cb);
    return;
  }
  Py_buffer v;
  if (PyObject_GetBuffer(data.ptr(), &v, PyBUF_ANY_CONTIGUOUS) < 0) throw_py();
  d.data = v.buf;
  d.len = (uint32_t)v.len;
  CkDynContribute(&me, &d, &cb);
  PyBuffer_Release(&v);
}

nb::list mod_reducers() {
  const CkDynReducerInfo *t = nullptr;
  int n = CkDynReducers(&t);
  nb::list l;
  for (int i = 0; i < n; i++) l.append(nb::make_tuple(nb::str(t[i].name), t[i].type, (int)t[i].elem));
  return l;
}

void mod_future_set(int pe, uint32_t fid, nb::handle value) {
  BuilderGuard bg(CkDynMsgBegin(2, 64, 0, 0, -1));
  Packer pk(bg.b);
  if (!pk.add(value.ptr()) || !pk.finish()) throw_py();
  CkDynMsg *m = bg.finish(0);
  CkDynFutureSet(pe, fid, m);
}

nb::tuple mod_topo_tree_edges(int pe, int root, nb::handle pes, int bfactor) {
  std::vector<int> pv;
  if (pes.is_none()) {
    int n = CkDynNumPes();
    pv.push_back(root);
    for (int i = 0; i < n; i++) if (i != root) pv.push_back(i);
  } else {
    pv = int_list(pes);
  }
  int parent = -1;
  std::vector<int> kids(pv.size() + 1);
  int nk = CkDynTopoTreeEdges(pe, root, pv.data(), (int)pv.size(), bfactor, &parent, kids.data(), (int)kids.size());
  nb::list children;
  for (int i = 0; i < nk && i < (int)kids.size(); i++) children.append(kids[i]);
  nb::object p = parent < 0 ? nb::none() : nb::object(nb::int_(parent));
  return nb::make_tuple(p, children);
}

void mod_exit(int code) {
  flush_stdio();
  PyThreadState *ts = PyEval_SaveThread();
  CkDynExit(code);
  PyEval_RestoreThread(ts);
}

}  // namespace

// ---------------------------------------------------------------------------
// Module

NB_MODULE(_ckdyn, m) {
  if (_import_array() < 0) throw nb::python_error();
  // Interpreter shutdown after a run is abrupt (CkExit ends the process) and
  // module-level objects can outlive nanobind's leak check.
  nb::set_leak_warnings(false);
  PyGreenlet_Import();
  if (!_PyGreenlet_API) throw nb::python_error();

  g_empty_tuple = PyTuple_New(0);
  g_DONE = PyUnicode_InternFromString("_ckdyn.DONE");
  g_EXC = PyUnicode_InternFromString("_ckdyn.EXC");
#define INTERN(var, s) var = PyUnicode_InternFromString(s)
  INTERN(s_cond_next, "_cond_next");
  INTERN(s_flush, "__flush_wait_queues__");
  INTERN(s_ckdyn_target, "_ckdyn_target");
  INTERN(s_src, "src");
  INTERN(s_fid, "fid");
  INTERN(s_store, "store");
  INTERN(s_append, "append");
  INTERN(s_frombytes, "frombytes");
  INTERN(s_ret, "ret");
  INTERN(s_awaitable, "awaitable");
  INTERN(s_is_ray, "is_ray");
  INTERN(s_Future, "Future");
  INTERN(s_createFuture, "createFuture");
  INTERN(s_name, "__name__");
  INTERN(k_is_ray, "is_ray");
#undef INTERN
  s_kw_protocol_cb = Py_BuildValue("(ss)", "protocol", "buffer_callback");
  s_kw_buffers = Py_BuildValue("(s)", "buffers");
  s_kw_store = Py_BuildValue("(s)", "store");
  k_block = PyBytes_FromString("block");
  k_bcast = PyBytes_FromString("bcast");
  k_bcastret = PyBytes_FromString("bcastret");
  k_creation = PyBytes_FromString("creation");
  k_custom_reducer = PyBytes_FromString("custom_reducer");
  k_single = PyBytes_FromString("single");

  nb::module_ pickle = nb::module_::import_("pickle");
  g_dumps = nb::object(pickle.attr("dumps")).release().ptr();
  g_loads = nb::object(pickle.attr("loads")).release().ptr();
  g_array_type = nb::object(nb::module_::import_("array").attr("array")).release().ptr();
  g_format_exception = nb::object(nb::module_::import_("traceback").attr("format_exception")).release().ptr();
  g_worker_main = PyCFunction_New(&worker_main_def, nullptr);

  nb::register_exception_translator([](const std::exception_ptr &p, void *) {
    try {
      std::rethrow_exception(p);
    } catch (const CkDynError &e) {
      PyObject *t = charm4py_error_type();
      PyErr_SetString(t, e.what());
      Py_DECREF(t);
    }
  });

  // raw types
  MsgRefType.tp_name = "charm4py._ckdyn.MsgRef";
  MsgRefType.tp_basicsize = sizeof(MsgRefObject);
  MsgRefType.tp_dealloc = msgref_dealloc;
  MsgRefType.tp_flags = Py_TPFLAGS_DEFAULT;
  MsgRefType.tp_doc = "Holds one retain of a delivered CkDyn message (base of received numpy views).";

  ProxyType.tp_name = "charm4py._ckdyn.Proxy";
  ProxyType.tp_basicsize = sizeof(ProxyObject);
  ProxyType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  ProxyType.tp_new = PyType_GenericNew;
  ProxyType.tp_init = proxy_init;
  ProxyType.tp_getset = proxy_getset;
  ProxyType.tp_methods = proxy_methods;
  ProxyType.tp_richcompare = proxy_richcompare;
  ProxyType.tp_hash = proxy_hash;
  ProxyType.tp_repr = proxy_repr;
  ProxyType.tp_doc = "Base class of every charm4py proxy class.";

  PMType.tp_name = "charm4py._ckdyn.ProxyMethod";
  PMType.tp_basicsize = sizeof(PMObject);
  PMType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_METHOD_DESCRIPTOR;
  PMType.tp_vectorcall_offset = offsetof(PMObject, vectorcall);
  PMType.tp_call = PyVectorcall_Call;
  PMType.tp_descr_get = pm_descr_get;
  PMType.tp_dealloc = pm_dealloc;
  PMType.tp_getset = pm_getset;
  PMType.tp_repr = pm_repr;

  BPMType.tp_name = "charm4py._ckdyn.BoundProxyMethod";
  BPMType.tp_basicsize = sizeof(BPMObject);
  BPMType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL;
  BPMType.tp_vectorcall_offset = offsetof(BPMObject, vectorcall);
  BPMType.tp_call = PyVectorcall_Call;
  BPMType.tp_dealloc = bpm_dealloc;
  BPMType.tp_getset = bpm_getset;
  BPMType.tp_methods = bpm_methods;
  BPMType.tp_richcompare = bpm_richcompare;
  BPMType.tp_hash = bpm_hash;
  BPMType.tp_repr = bpm_repr;

  for (PyTypeObject *t : {&MsgRefType, &ProxyType, &PMType, &BPMType})
    if (PyType_Ready(t) < 0) throw nb::python_error();
  m.attr("MsgRef") = nb::borrow((PyObject *)&MsgRefType);
  m.attr("Proxy") = nb::borrow((PyObject *)&ProxyType);
  m.attr("ProxyMethod") = nb::borrow((PyObject *)&PMType);
  m.attr("BoundProxyMethod") = nb::borrow((PyObject *)&BPMType);

  nb::object modname = m.attr("__name__");
  for (PyMethodDef *d = raw_functions; d->ml_name; d++) {
    PyObject *f = PyCFunction_NewEx(d, nullptr, modname.ptr());
    if (!f) throw nb::python_error();
    m.attr(d->ml_name) = nb::steal(f);
  }
  g_rebuild_bound = m.attr("_rebuild_bound").ptr();
  Py_INCREF(g_rebuild_bound);

  // Target
  nb::class_<TargetW>(m, "Target")
      .def_prop_ro("kind", [](const TargetW &t) { return (int)t.t.kind; })
      .def_prop_ro("chare_idx", [](const TargetW &t) { return t.t.chare_idx; })
      .def_prop_ro("gid", [](const TargetW &t) -> nb::object {
        if (!is_group_kind(t.t.kind)) throw nb::attribute_error("gid");
        return nb::int_(t.t.u.group.gid);
      })
      .def_prop_ro("aid", [](const TargetW &t) -> nb::object {
        if (t.t.kind != CKDYN_ARRAY) throw nb::attribute_error("aid");
        return nb::int_(t.t.u.array.aid);
      })
      .def_prop_ro("pe", [](const TargetW &t) {
        return is_group_kind(t.t.kind) ? t.t.u.group.pe_or_node : -1;
      })
      .def_prop_ro("index", [](const TargetW &t) -> nb::object {
        if (t.t.kind != CKDYN_ARRAY) return nb::none();
        return nb::steal(index_tuple(t.t.u.array.index, t.t.u.array.ndims));
      })
      .def_prop_ro("cid", [](const TargetW &t) -> nb::object {
        if (t.t.kind != CKDYN_CHARE && t.t.kind != CKDYN_MAINCHARE) return nb::none();
        return nb::make_tuple(t.t.u.chare.pe, (unsigned long long)(uintptr_t)t.t.u.chare.ptr);
      })
      .def("__getstate__", [](const TargetW &t) { return nb::bytes((const char *)&t.t, sizeof t.t); })
      .def("__setstate__", [](TargetW &t, nb::bytes b) {
        if (b.size() != sizeof(CkDynTarget)) throw nb::value_error("Target state of the wrong size");
        new (&t) TargetW();
        memcpy(&t.t, b.c_str(), sizeof t.t);
      })
      .def("__eq__", [](const TargetW &a, nb::handle b) -> nb::object {
        TargetW *o = nullptr;
        if (!nb::try_cast<TargetW *>(b, o, false) || !o) return nb::borrow(Py_NotImplemented);
        return nb::bool_(a.t.chare_idx == o->t.chare_idx && target_eq(a.t, o->t));
      })
      .def("__hash__", [](const TargetW &a) { return (Py_ssize_t)target_hash(a.t); })
      .def("__repr__", [](const TargetW &t) {
        char buf[160];
        const CkDynTarget &x = t.t;
        if (x.kind == CKDYN_ARRAY) {
          std::string idx;
          for (int i = 0; i < x.u.array.ndims; i++) idx += (i ? "," : "") + std::to_string(x.u.array.index[i]);
          snprintf(buf, sizeof buf, "Target(ARRAY, chare_idx=%d, aid=%d, index=(%s))", x.chare_idx, x.u.array.aid, idx.c_str());
        } else if (is_group_kind(x.kind)) {
          snprintf(buf, sizeof buf, "Target(kind=%d, chare_idx=%d, gid=%d, pe=%d)", (int)x.kind, x.chare_idx, x.u.group.gid, x.u.group.pe_or_node);
        } else {
          snprintf(buf, sizeof buf, "Target(kind=%d, chare_idx=%d, pe=%d, ptr=%p)", (int)x.kind, x.chare_idx, x.u.chare.pe, x.u.chare.ptr);
        }
        return std::string(buf);
      });

  // Callback
  nb::class_<CallbackW>(m, "Callback")
      .def_static("target", [](nb::handle bpm, uint32_t refnum) {
        if (!PyObject_TypeCheck(bpm.ptr(), &BPMType)) throw nb::type_error("Callback.target: expected a bound proxy method");
        CallbackW w{callback_from(bpm)};
        w.cb.refnum = refnum;
        return w;
      }, nb::arg("bpm"), nb::arg("refnum") = 0)
      .def_static("future", [](int pe, uint32_t fid) {
        CallbackW w; memset(&w.cb, 0, sizeof w.cb);
        w.cb.kind = CKDYN_CB_FUTURE; w.cb.pe = pe; w.cb.id = fid;
        return w;
      })
      .def_static("ignore", []() { CallbackW w; memset(&w.cb, 0, sizeof w.cb); w.cb.kind = CKDYN_CB_IGNORE; return w; })
      .def_static("exit", []() { CallbackW w; memset(&w.cb, 0, sizeof w.cb); w.cb.kind = CKDYN_CB_EXIT; return w; })
      .def_prop_ro("kind", [](const CallbackW &w) { return (int)w.cb.kind; })
      .def_prop_ro("ep", [](const CallbackW &w) { return w.cb.ep; })
      .def_prop_ro("bcast", [](const CallbackW &w) { return w.cb.bcast != 0; })
      .def_prop_ro("pe", [](const CallbackW &w) { return w.cb.pe; })
      .def_prop_ro("id", [](const CallbackW &w) { return w.cb.id; })
      .def_prop_ro("refnum", [](const CallbackW &w) { return w.cb.refnum; })
      .def_prop_ro("target_", [](const CallbackW &w) { return nb::steal(make_target(w.cb.t)); })
      .def("__getstate__", [](const CallbackW &w) { return nb::bytes((const char *)&w.cb, sizeof w.cb); })
      .def("__setstate__", [](CallbackW &w, nb::bytes b) {
        if (b.size() != sizeof(CkDynCallback)) throw nb::value_error("Callback state of the wrong size");
        new (&w) CallbackW();
        memcpy(&w.cb, b.c_str(), sizeof w.cb);
      });

  // constants
  m.attr("MAINCHARE") = (int)CKDYN_MAINCHARE;
  m.attr("CHARE") = (int)CKDYN_CHARE;
  m.attr("GROUP") = (int)CKDYN_GROUP;
  m.attr("NODEGROUP") = (int)CKDYN_NODEGROUP;
  m.attr("ARRAY") = (int)CKDYN_ARRAY;
  m.attr("ARRAYMAP") = (int)CKDYN_ARRAYMAP;
  m.attr("EP_THREADED") = (int)CKDYN_EP_THREADED;
  m.attr("EP_INLINE") = (int)CKDYN_EP_INLINE;
  m.attr("EP_EXPEDITED") = (int)CKDYN_EP_EXPEDITED;
  m.attr("EP_IMMEDIATE") = (int)CKDYN_EP_IMMEDIATE;
  m.attr("EP_NOTRACE") = (int)CKDYN_EP_NOTRACE;
  m.attr("EP_APPWORK") = (int)CKDYN_EP_APPWORK;
  m.attr("EP_EXCLUSIVE") = (int)CKDYN_EP_EXCLUSIVE;
  m.attr("EP_REDNTARGET") = (int)CKDYN_EP_REDNTARGET;
  m.attr("MSG_BLOCK") = (int)CKDYN_MSG_BLOCK;
  m.attr("MSG_BCAST") = (int)CKDYN_MSG_BCAST;
  m.attr("MSG_BCASTRET") = (int)CKDYN_MSG_BCASTRET;
  m.attr("MSG_CREATION") = (int)CKDYN_MSG_CREATION;
  m.attr("MSG_USESATSYNC") = (int)CKDYN_MSG_USESATSYNC;
  m.attr("MSG_RAY") = (int)CKDYN_MSG_RAY;
  m.attr("MSG_CONSTRAINED") = (int)CKDYN_MSG_CONSTRAINED;
  m.attr("PYMSG_REDRESULT") = (int)PYMSG_REDRESULT;
  m.attr("PYMSG_SINGLE") = (int)PYMSG_SINGLE;
  m.attr("PUP_MIGRATE") = (int)CKDYN_PUP_MIGRATE;
  m.attr("PUP_CHECKPOINT") = (int)CKDYN_PUP_CHECKPOINT;
  m.attr("PE_ANY") = (int)CKDYN_PE_ANY;
  m.attr("CB_TARGET") = (int)CKDYN_CB_TARGET;
  m.attr("CB_FUTURE") = (int)CKDYN_CB_FUTURE;
  m.attr("CB_IGNORE") = (int)CKDYN_CB_IGNORE;
  m.attr("CB_EXIT") = (int)CKDYN_CB_EXIT;
  m.attr("ELEM_I8") = (int)CKDYN_ELEM_I8;
  m.attr("ELEM_I16") = (int)CKDYN_ELEM_I16;
  m.attr("ELEM_I32") = (int)CKDYN_ELEM_I32;
  m.attr("ELEM_I64") = (int)CKDYN_ELEM_I64;
  m.attr("ELEM_U8") = (int)CKDYN_ELEM_U8;
  m.attr("ELEM_U16") = (int)CKDYN_ELEM_U16;
  m.attr("ELEM_U32") = (int)CKDYN_ELEM_U32;
  m.attr("ELEM_U64") = (int)CKDYN_ELEM_U64;
  m.attr("ELEM_F16") = (int)CKDYN_ELEM_F16;
  m.attr("ELEM_F32") = (int)CKDYN_ELEM_F32;
  m.attr("ELEM_F64") = (int)CKDYN_ELEM_F64;
  m.attr("ELEM_C64") = (int)CKDYN_ELEM_C64;
  m.attr("ELEM_C128") = (int)CKDYN_ELEM_C128;
  m.attr("ELEM_BOOL") = (int)CKDYN_ELEM_BOOL;
  m.attr("ELEM_BYTE") = (int)CKDYN_ELEM_BYTE;

  // functions
  m.def("init", &mod_init, nb::arg("charm"), nb::arg("options"), nb::arg("future_type").none() = nb::none(),
        nb::arg("remote_future").none() = nb::none(), nb::arg("collective_future_type").none() = nb::none());
  m.def("start", &mod_start, nb::arg("argv"));
  m.def("register_class", &mod_register_class, nb::arg("cls"), nb::arg("kind"), nb::arg("ndims"), nb::arg("entries"));
  m.def("set_proxy_class", &mod_set_proxy_class);
  m.def("api_version", []() { return CkDynApiVersion(); });
  m.def("refnum_bytes", []() { return CkDynRefnumBytes(); });
  m.def("has_device_support", []() { return CkDynHasDeviceSupport() != 0; });
  m.def("my_pe", []() { return CkDynMyPe(); });
  m.def("num_pes", []() { return CkDynNumPes(); });
  m.def("my_node", []() { return CkDynMyNode(); });
  m.def("num_nodes", []() { return CkDynNumNodes(); });
  m.def("my_rank", []() { return CkDynMyRank(); });
  m.def("node_size", [](int node) { return CkDynNodeSize(node); });
  m.def("node_of", [](int pe) { return CkDynNodeOf(pe); });
  m.def("wall_timer", []() { return CkDynWallTimer(); });
  m.def("exit", &mod_exit, nb::arg("code") = 0);
  m.def("abort", [](const std::string &msg) { flush_stdio(); CkDynAbort(msg.c_str()); });
  m.def("printf", [](const std::string &text) { CkDynPrintf(text.c_str()); });
  m.def("num_physical_nodes", []() { return CkDynNumPhysicalNodes(); });
  m.def("physical_node_id", [](int pe) { return CkDynPhysicalNodeID(pe); });
  m.def("physical_rank", [](int pe) { return CkDynPhysicalRank(pe); });
  m.def("num_pes_on_physical_node", [](int node) { return CkDynNumPesOnPhysicalNode(node); });
  m.def("pes_on_physical_node", [](int node) {
    int n = CkDynNumPesOnPhysicalNode(node);
    std::vector<int> v((size_t)(n > 0 ? n : 0) + 1);
    int k = CkDynGetPesOnPhysicalNode(node, v.data(), (int)v.size());
    nb::list l;
    for (int i = 0; i < k && i < (int)v.size(); i++) l.append(v[i]);
    return l;
  });
  m.def("node_first", [](int node) { return CkDynNodeFirst(node); });
  m.def("topo_tree_edges", &mod_topo_tree_edges, nb::arg("pe"), nb::arg("root"), nb::arg("pes").none(), nb::arg("bfactor"));
  m.def("make_proxy_method", &mod_make_proxy_method);
  m.def("send", [](nb::handle proxy, int ep, nb::handle args, int flags, nb::handle future) {
    PyObject *r = mod_send(proxy, ep, args, flags, future);
    Py_XDECREF(r);
  }, nb::arg("proxy"), nb::arg("ep"), nb::arg("args").none(), nb::arg("flags") = 0, nb::arg("future").none() = nb::none());
  m.def("create_chare", &mod_create_chare, nb::arg("chare_idx"), nb::arg("on_pe"), nb::arg("args").none(),
        nb::arg("flags") = 0, nb::arg("future").none() = nb::none());
  m.def("create_group", &mod_create_group, nb::arg("chare_idx"), nb::arg("args").none(), nb::arg("flags") = 0,
        nb::arg("future").none() = nb::none(), nb::arg("pes").none() = nb::none());
  m.def("create_nodegroup", &mod_create_nodegroup, nb::arg("chare_idx"), nb::arg("args").none(), nb::arg("flags") = 0,
        nb::arg("future").none() = nb::none());
  m.def("create_array", &mod_create_array, nb::arg("chare_idx"), nb::arg("dims").none(), nb::arg("ndims"), nb::arg("args").none(),
        nb::arg("map_gid") = -1, nb::arg("use_at_sync") = false, nb::arg("flags") = 0, nb::arg("future").none() = nb::none());
  m.def("array_insert", &mod_array_insert, nb::arg("aid"), nb::arg("index"), nb::arg("on_pe"), nb::arg("args").none(),
        nb::arg("flags") = 0, nb::arg("future").none() = nb::none(), nb::arg("use_at_sync") = false, nb::arg("chare_idx") = -1);
  m.def("array_done_inserting", [](int aid) { CkDynArrayDoneInserting(aid); });
  m.def("destroy_element", &mod_destroy_element);
  m.def("contribute", &mod_contribute, nb::arg("obj"), nb::arg("data").none(), nb::arg("reducer_type"), nb::arg("target"));
  m.def("reducers", &mod_reducers);
  m.def("foreign_reducer_type", []() { return CkDynForeignReducerType(); });
  m.def("reduction_number", [](nb::handle obj) { CkDynTarget t = target_of(obj.ptr()); return CkDynGetReductionNumber(&t); });
  m.def("future_set", &mod_future_set, nb::arg("pe"), nb::arg("fid"), nb::arg("value").none());
  m.def("start_qd", [](nb::handle target) { CkDynCallback cb = callback_from(target); CkDynStartQD(&cb); });
  m.def("at_sync", [](nb::handle obj) { CkDynTarget t = target_of(obj.ptr()); CkDynAtSync(&t); });
  m.def("set_migratable", [](nb::handle obj, bool flag) { CkDynTarget t = target_of(obj.ptr()); CkDynSetMigratable(&t, flag ? 1 : 0); });
  m.def("migrate", [](nb::handle obj, int pe) { CkDynTarget t = target_of(obj.ptr()); CkDynMigrateMe(&t, pe); });
  m.def("set_obj_load", [](nb::handle obj, double secs) { CkDynTarget t = target_of(obj.ptr()); CkDynSetObjLoad(&t, secs); });
  m.def("lb_instrument", [](bool on) { CkDynLBInstrument(on ? 1 : 0); });
  m.def("has_lb", []() { return CkDynHasLB() != 0; });
  m.def("callstack_push", [](nb::handle obj) { CkDynTarget t = target_of(obj.ptr()); CkDynCallstackPush(&t); });
  m.def("callstack_pop", [](nb::handle obj) { CkDynTarget t = target_of(obj.ptr()); CkDynCallstackPop(&t); });
  m.def("checkpoint", [](const std::string &dir, nb::handle target) {
    CkDynCallback cb = callback_from(target);
    CkDynCheckpoint(dir.c_str(), &cb);
  });
  m.def("main_greenlet", []() -> nb::object {
    if (!tls.main_gr) return nb::none();
    return nb::borrow(tls.main_gr);
  });
  m.def("last_ep", []() { return tls.last_ep; });
  m.def("trace_register_user_event", [](const std::string &desc, int id) {
    return CkDynTraceRegisterUserEvent(strdup(desc.c_str()), id);
  });
  m.def("trace_begin_user_bracket", [](int id) { CkDynTraceBeginUserBracketEvent(id); });
  m.def("trace_end_user_bracket", [](int id) { CkDynTraceEndUserBracketEvent(id); });
  m.def("ccs_register", [](const std::string &name) { CkDynCcsRegister(name.c_str(), ccs_handler); });
  m.def("ccs_is_remote_request", []() { return CkDynCcsIsRemoteRequest() != 0; });
  m.def("ccs_send_reply", [](nb::bytes b) { CkDynCcsSendReply((int)b.size(), b.c_str()); });
  m.def("ccs_delay_reply", []() { return (uintptr_t)CkDynCcsDelayReply(); });
  m.def("ccs_send_delayed_reply", [](uintptr_t token, nb::bytes b) {
    CkDynCcsSendDelayedReply((void *)token, (int)b.size(), b.c_str());
  });
  m.def("call_after", [](int tag, double msecs) { CkDynCallAfter(call_after_fn, (void *)(intptr_t)tag, msecs); });
  m.def("hapi_add_callback", [](uint64_t stream, nb::handle future) {
    if (!CkDynHasDeviceSupport()) throw CkDynError("hapi_add_callback: this libcharm has no device support");
    CkDynCallback cb = callback_from(future);
    CkDynHapiAddCallback(stream, &cb);
  });
  m.def("is_restarting", []() { return CkDynIsRestarting() != 0; });

  init_objstore(m);
}
