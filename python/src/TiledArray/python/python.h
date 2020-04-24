#ifndef TA_PYTHON_H
#define TA_PYTHON_H

#pragma GCC diagnostic ignored "-Wregister"

#include <TiledArray/util/vector.h>
#include <TiledArray/size_array.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifndef TA_PYTHON_MAX_EXPRESSION
#define TA_PYTHON_MAX_EXPRESSION 5
#endif

#if TA_PYTHON_MAX_EXPRESSION < 1
#error "TA_PYTHON_MAX_EXPRESSION must be > 0"
#endif

namespace py = pybind11;

namespace pybind11 {
namespace detail {

  template <typename T, std::size_t N>
  struct type_caster< TiledArray::container::svector<T,N> >
    : list_caster< TiledArray::container::svector<T,N>, T > { };

  template <typename T>
  struct type_caster< TiledArray::detail::SizeArray<T> >
    : list_caster< TiledArray::detail::SizeArray<T>, T > { };

}
}

#endif // TA_PYTHON_H
