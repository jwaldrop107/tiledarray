#include "TiledArray/python/module.h"

PYBIND11_MODULE(tiledarray, m) {
  TiledArray::python::initialize(m);
}
