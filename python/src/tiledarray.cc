#include "TiledArray/python/module.h"

PYBIND11_MODULE(tiledarray, m) {

  TiledArray::python::initialize(m);

  auto atexit = py::module::import("atexit");
  atexit.attr("register")(
    py::cpp_function([]() {
        TiledArray::python::finalize();
      }
    )
  );

}
