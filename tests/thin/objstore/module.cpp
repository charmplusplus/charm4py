#include <nanobind/nanobind.h>
namespace nb = nanobind;
void init_objstore(nb::module_ &m);
NB_MODULE(_objstore_test, m) { init_objstore(m); }
