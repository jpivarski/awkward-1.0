// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#include <string>

#include <pybind11/pybind11.h>

#include "awkward/python/content.h"
#include "awkward/python/io.h"
#include "awkward/python/forth.h"

namespace py = pybind11;
PYBIND11_MODULE(_ext, m, py::mod_gil_not_used()) {
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  ////////// content.h

  make_ArrayBuilder(m, "ArrayBuilder");

  ////////// io.h

  make_fromjsonobj(m, "fromjsonobj");
  make_fromjsonobj_schema(m, "fromjsonobj_schema");

  ////////// forth.h

  make_ForthMachineOf<int32_t, int32_t>(m, "ForthMachine32");
  make_ForthMachineOf<int64_t, int32_t>(m, "ForthMachine64");

}
