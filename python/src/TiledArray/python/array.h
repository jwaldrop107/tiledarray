#ifndef TA_PYTHON_ARRAY_H
#define TA_PYTHON_ARRAY_H

#include "python.h"
#include "expression.h"
#include "trange.h"
#include "range.h"

#include <TiledArray/dist_array.h>
#include <vector>
#include <string>

namespace TiledArray {
namespace python {
namespace array {

  // template<typename T>
  // py::array_t<T> make_tile(Tensor<T> &tile) {
  //   auto buffer_info = make_buffer_info(tile);
  //   return py::array_t<T>(
  //     buffer_info.shape,
  //     buffer_info.strides,
  //     (T*)buffer_info.ptr,
  //     py::cast(tile)
  //   );
  // }

  template<typename T>
  auto make_tile(py::buffer data) {
    auto shape = data.request().shape;
    py::array_t<T> tmp(shape);
    int result = py::detail::npy_api::get().PyArray_CopyInto_(tmp.ptr(), data.ptr());
    if (result < 0) throw py::error_already_set();
    return Tensor<T>(Range(shape), tmp.data());
  }

  // std::function<py::buffer(const Range&)>
  void init_tiles(TArray<double> &a, py::object f) {
    py::gil_scoped_release gil;
    auto op = [f](const Range& range) {
      Tensor<double> tile;
      {
        py::gil_scoped_acquire acquire;
        //py::print(f);
        py::buffer buffer = f(range);
        tile = make_tile<double>(buffer);
      }
      return tile;
    };
    a.init_tiles(op);
    a.world().gop.fence();
  }

  template<class ... Trange>
  std::shared_ptr< TArray<double> > make_array(const Trange& ... args, World *world, py::object op) {
    if (!world) {
      world = &get_default_world();
    }
    auto array = std::make_shared< TArray<double> >(*world, trange::make_trange(args...));
    if (!op.is_none()) {
      init_tiles(*array, op);
    }
    return array;
  }

  inline std::vector<size_t> shape(const TArray<double> &a) {
    std::vector<size_t> shape;
    for (const auto &tr1 : a.trange().data()) {
      shape.push_back(tr1.extent());
    }
    return shape;
  }

  inline auto world(const TArray<double> &a) {
    return &a.world();
  }

  inline auto trange(const TArray<double> &a) {
    return trange::list(a.trange());
  }

  template<typename T>
  py::buffer_info make_buffer_info(Tensor<T> &tile) {
    std::vector<size_t> strides;
    for (auto s : tile.range().stride()) {
      strides.push_back(sizeof(T)*s);
    }
    return py::buffer_info(
      tile.data(),                              /* Pointer to buffer */
      sizeof(T),                                /* Size of one scalar */
      py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
      tile.range().rank(),                                        /* Number of dimensions */
      tile.range().extent(),                             /* Buffer dimensions */
      strides                            /* Strides (in bytes) for each index */
    );
  }

  // template<class Array>
  // struct Iterator {
  //   std::shared_ptr<Array> array;
  //   typedef typename Array::iterator iterator;
  //   auto operator++() {
  //     return ++iterator;
  //   }
  //   auto operator*() {
  //     auto index = iterator.index();
  //     return std::make_tuple(
  //       std::vector<int64_t>(index.begin(), index.end()),
  //       py::array(
  //       )
  //     );
  //   }
  //   bool operator==(Iterator other) const {
  //     return this->it == other.it;
  //   }
  // };

  inline auto make_iterator(TArray<double> &array) {
    return py::make_iterator(array.begin(), array.end());
  }

  inline void setitem(TArray<double> &array, std::vector<int64_t> idx, py::buffer data) {
    auto tile = make_tile<double>(data);
    array.set(idx, tile);
  }

  inline py::array getitem(const TArray<double> &array, std::vector<int64_t> idx) {
    auto tile = array.find(idx).get();
    return py::array(make_buffer_info(tile));
  }

  typedef TArray<double>::reference TileReference;

  py::array get_reference_data(TileReference &r) {
    auto tile = r.get();
    auto shape = tile.range().extent();
    auto base = py::cast(r);
    return py::array_t<double>(shape, tile.data(), base);
  }

  void set_reference_data(TileReference &r, py::buffer data) {
    r = make_tile<double>(data);
  }

  void __init__(py::module m) {

    py::class_< TArray<double>::reference >(m, "TileReference", py::module_local())
      .def_property_readonly("index", &TileReference::index)
      .def_property_readonly("range", &TileReference::make_range)
      .def_property("data", &get_reference_data, &set_reference_data)
      ;

    py::class_< TArray<double>, std::shared_ptr<TArray<double> > >(m, "TArray")
      .def(py::init())
      .def(
        py::init(&make_array< std::vector<int64_t>, size_t >),
        py::arg("shape"),
        py::arg("block"),
        py::arg("world") = nullptr,
        py::arg("op") = py::none()
      )
      .def(
        py::init(&array::make_array< std::vector< std::vector<int64_t> > >),
        py::arg("trange"),
        py::arg("world") = nullptr,
        py::arg("op") = py::none()
      )
      .def_property_readonly("world", &array::world, py::return_value_policy::reference)
      .def_property_readonly("trange", &array::trange)
      .def_property_readonly("shape", &array::shape)
      .def("fill", &TArray<double>::fill, py::arg("value"), py::arg("skip_set") = false)
      .def("init", &array::init_tiles)
      .def("__iter__", &array::make_iterator, py::keep_alive<0, 1>()) // Keep object alive while iterator is used */
      .def("__getitem__", &expression::getitem)
      .def("__setitem__", &expression::setitem)
      .def("__getitem__", &array::getitem)
      .def("__setitem__", &array::setitem)
      ;

    // py::class_< Tensor<double>, std::shared_ptr<Tensor<double> > >(m, "Tensor",  py::buffer_protocol())
    //   .def_buffer(&array::make_buffer_info<double>)
    //   ;

  }


}
}
}

#endif // TA_PYTHON_ARRAY_H
