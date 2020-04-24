#include "python.h"
#include "range.h"
#include "trange.h"
#include "array.h"
#include "expression.h"
#include "einsum.h"

#include <tiledarray.h>
#include <dlfcn.h>

namespace TiledArray {
namespace python {

  static World& initialize() {

#if defined( __linux__) && defined(OPEN_MPI)
    dlopen("libmpi.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL); // ompi hack
#endif

    // this loads MPI before TA tries to do it
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized) {
      // Initialize runtime
      int argc = 0;
      char *_argv[0];
      char **argv = _argv;
      //MPI_Init(&argc, &argv);
      TiledArray::World& world = TiledArray::initialize(argc, argv);
      if (world.rank() == 0) {
        std::cout << "initialized TA in a world with " << world.size() << " ranks" << std::endl;
      }
    }

    return get_default_world();

  }

  void finalize() {
    if (!TiledArray::finalized()) {
      TiledArray::finalize();
    }
  }

  py::object default_world() {
    return py::cast(TiledArray::get_default_world(), py::return_value_policy::reference);
  }

  World& initialize(py::module m) {

    auto &world = initialize();

    py::class_< TiledArray::World >(m, "World")
      // .def_property_readonly_static(
      //   "COMM_WORLD",
      //   [](){ return madness::World::find_instance(SafeMPI::COMM_WORLD); }
      // )
      .def_property_readonly("rank", &TiledArray::World::rank)
      .def_property_readonly("size", &TiledArray::World::size)
      .def("fence", [](TiledArray::World &world) { world.gop.fence(); })
      ;

    m.def("get_default_world", &default_world);

    m.def("finalize", &TiledArray::python::finalize);

    range::__init__(m);
    trange::__init__(m);
    expression::__init__(m);
    array::__init__(m);
    einsum::__init__(m);

    return world;

  }

}
}
