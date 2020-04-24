#ifndef TA_PYTHON_TRANGE_H
#define TA_PYTHON_TRANGE_H

#include "python.h"

#include <TiledArray/tiled_range.h>
#include <vector>
#include <string>

namespace TiledArray {
namespace python {
namespace trange {

  // template<class ... Args>
  // inline TiledRange make_trange(Args ... args);

  auto list(const TiledRange &trange) {
    std::vector< std::vector<size_t> > v;
    for (auto tr1 : trange.data()) {
      auto it = tr1.begin();
      v.push_back({it->first});
      for (; it != tr1.end(); ++it) {
        v.back().push_back(it->second);
      }
    }
    return v;
  }

  //template<>
  inline TiledRange make_trange(std::vector< std::vector<int64_t> > trange) {
    std::vector<TiledRange1> trange1;
    for (auto tr : trange) {
      trange1.emplace_back(tr.begin(), tr.end());
    }
    return TiledRange(trange1.begin(), trange1.end());
  }

  //template<>
  inline TiledRange make_trange(std::vector<int64_t> shape, size_t block) {
    std::vector<TiledRange1> trange1;
    for (size_t i = 0; i < shape.size(); ++i) {
      std::vector<int64_t> tr1;
      for (size_t j = 0; j <= (shape[i] + block-1); j += block) {
        tr1.push_back(std::min<int64_t>(j,shape[i]));
      }
      trange1.push_back(TiledRange1(tr1.begin(), tr1.end()));
    }
    return TiledRange(trange1.begin(), trange1.end());
  }

  void __init__(py::module m) {

    // py::class_<TiledRange>(m, "TiledRange")
    //   .def(py::init(&make_trange< std::vector< std::vector<int64_t> > >))
    //   ;

    // py::implicitly_convertible< std::vector< std::vector<int64_t> >, TiledRange>();

  }


}
}
}

#endif // TA_PYTHON_TRANGE_H
