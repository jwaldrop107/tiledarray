#ifndef TILEDARRAY_CONTRACTION_H__INCLUDED
#define TILEDARRAY_CONTRACTION_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/range.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/bitset.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/cblas.h>
#include <world/array.h>
#include <iterator>
#include <numeric>
#include <functional>


namespace TiledArray {
  namespace math {

    namespace {
      /// Contraction type selection for complex numbers

      /// \tparam T The left contraction argument type
      /// \tparam U The right contraction argument type
      template <typename T, typename U>
      struct ContractionValue {
        typedef T type; ///< The result type
      };

      template <typename T>
      struct ContractionValue<T, std::complex<T> > {
        typedef std::complex<T> type;
      };

      template <typename T>
      struct ContractionValue<std::complex<T>, T> {
        typedef std::complex<T> type;
      };

    } // namespace

    /// A utility class for contraction operations.

    /// \param I The index type
    /// \note All algorithms assume the object dimension layout is in the same
    /// order as the left- and right-hand variable list arguments used to
    /// construct this object.
    class Contraction {
    public:
      typedef std::vector<std::size_t> map_type;

      Contraction(const expressions::VariableList& left,
          const expressions::VariableList& right, const detail::DimensionOrderType& order)
      {

        // Reserve storage for maps.
        left_inner_.reserve(left.dim());
        left_outer_.reserve(left.dim());
        right_inner_.reserve(right.dim());
        right_outer_.reserve(right.dim());

        // construct the inner and outer maps.
        for(expressions::VariableList::const_iterator it = left.begin(); it != left.end(); ++it) {
          expressions::VariableList::const_iterator right_it =
              std::find(right.begin(), right.end(), *it);
          if(right_it == right.end())
            left_outer_.push_back(std::distance(left.begin(), it));
          else {
            left_inner_.push_back(std::distance(left.begin(), it));
            right_inner_.push_back(std::distance(right.begin(), right_it));
          }
        }

        for(expressions::VariableList::const_iterator it = right.begin(); it != right.end(); ++it) {
          expressions::VariableList::const_iterator left_it =
              std::find(left.begin(), left.end(), *it);
          if(left_it == left.end())
            right_outer_.push_back(std::distance(right.begin(), it));
        }

        init_permutation(left, left_inner_, left_outer_, order, perm_left_, do_perm_left_);
        init_permutation(right, right_inner_, right_outer_, order, perm_right_, do_perm_right_);
      }



      const Permutation& perm_left() const { return perm_left_; }
      const Permutation& perm_right() const { return perm_right_; }



      /// Contract array

      template <typename ResArray, typename LeftArray, typename RightArray>
      void contract_array(ResArray& res, const LeftArray& left, const RightArray& right) const {
        TA_ASSERT(left.size() == left_dim());
        TA_ASSERT(right.size() == right_dim());
        TA_ASSERT(res.size() == dim());

        typename ResArray::iterator res_it = res.begin();

        for(map_type::const_iterator it = left_outer_.begin(); it != left_outer_.end(); ++it)
          *res_it++ = left[*it];
        for(map_type::const_iterator it = right_outer_.begin(); it != right_outer_.end(); ++it)
          *res_it++ = right[*it];
      }

      expressions::VariableList contract_vars(const expressions::VariableList& left, const expressions::VariableList& right) {
        std::vector<std::string> res(dim());
        contract_array(res, left.data(), right.data());
        return expressions::VariableList(res.begin(),res.end());
      }

      /// Generate a contracted range object

      /// \tparam LeftRange The left-hand range type.
      /// \tparam RightRange The right-hand range type.
      /// \param left The left-hand-argument range
      /// \param right The right-hand-argument range
      /// \return A contracted range object
      /// \throw TiledArray::Exception When the order of the ranges do not match
      /// \throw TiledArray::Exception When the dimension of \c left and \c right
      /// do not match the dimensions of the variable lists used to construct
      /// this object.
      template <typename LeftRange, typename RightRange>
      DynamicRange contract_range(const LeftRange& left, const RightRange& right) const {
        TA_ASSERT(left.order() == right.order());
        TA_ASSERT(left.dim() == left_dim());
        TA_ASSERT(right.dim() == right_dim());

        typename DynamicRange::index start(dim()), finish(dim());
        contract_array(start, left.start(), right.start());
        contract_array(finish, left.finish(), right.finish());
        return DynamicRange(start, finish, left.order());
      }

      template <typename LeftTRange, typename RightTRange>
      DynamicTiledRange contract_trange(const LeftTRange& left, const RightTRange& right) const {
        TA_ASSERT(left.tiles().order() == right.tiles().order());
        typename DynamicTiledRange::Ranges ranges(dim());
        contract_array(ranges, left.data(), right.data());
        return DynamicTiledRange(ranges.begin(), ranges.end(), left.tiles().order());
      }

      // Definition is in contraction_tensor.h
      template <typename LeftTensor, typename RightTensor>
      inline detail::Bitset<> contract_shape(const LeftTensor& left, const RightTensor& right);

      /// Permute tensor contraction

      /// This function permutes the left- and right-hand tensor arguments such
      /// that the contraction can be performed with a matrix-matrix
      /// multiplication. The permutation applied to \c left and \c right are
      /// defined by the variable lists used to construct this object. The data
      /// layout will match the expected input of the \c contract_tensor()
      /// function. The resulting variable list can be obtained with the
      /// \c contract_vars() function.
      /// \tparam Left The left-hand-side tensor argument type
      /// \tparam Right The right-hand-side tensor argument type
      /// \param left The left-hand-side tensor argument
      /// \param right The right-hand-side tensor argument
      /// \throw TiledArray::Exception When the orders of left and right are not
      /// equal
      template <typename Left, typename Right>
      expressions::Tensor<typename ContractionValue<typename Left::value_type,
          typename Right::value_type>::type, DynamicRange>
      permute_contract_tensor(const Left& left, const Right& right) const {
        TA_ASSERT(left.range().order() == right.range().order());
        TA_ASSERT(left.range().dim() == left_dim());
        TA_ASSERT(right.range().dim() == right_dim());

        if(do_perm_left_) {
          if(do_perm_right_) {
            return contract_tensor(
                expressions::make_permute_tensor(left, perm_left_).eval(),
                expressions::make_permute_tensor(right, perm_right_).eval());
          } else {
            return contract_tensor(
                expressions::make_permute_tensor(left, perm_left_).eval(),
                right.eval());
          }
        } else {
          if(do_perm_right_) {
            return contract_tensor(left.eval(),
                expressions::make_permute_tensor(right, perm_right_).eval());
          } else {
            return contract_tensor(left.eval(), right.eval());
          }
        }
      }

      /// Calculate the outer dimension for the left argument

      /// Assume outer dimensions are all on the left for decreasing
      /// dimension ordering and to the right for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused outer dimensions for the left argument
      template <typename D>
      std::size_t left_outer(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin(), range.size().begin() + left_outer_dim());
        else
          return accumulate(range.size().begin() + left_inner_dim(), range.size().end());
      }

      /// Calculate the inner dimension for the left argument

      /// Assume inner dimensions are all on the right for decreasing
      /// dimension ordering and to the left for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused inner dimensions for the left argument
      template <typename D>
      std::size_t left_inner(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin() + left_outer_dim(), range.size().end());
        else
          return accumulate(range.size().begin(), range.size().begin() + left_inner_dim());
      }

      /// Calculate the outer dimension for the right argument

      /// Assume outer dimensions are all on the left for decreasing
      /// dimension ordering and to the right for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused outer dimensions for the right argument
      template <typename D>
      std::size_t right_outer(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin(), range.size().begin() + right_outer_dim());
        else
          return accumulate(range.size().begin() + right_inner_dim(), range.size().end());
      }

      /// Calculate the inner dimension for the right argument

      /// Assume inner dimensions are all on the right for decreasing
      /// dimension ordering and to the left for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused inner dimensions for the right argument
      template <typename D>
      std::size_t right_inner(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin() + right_outer_dim(), range.size().end());
        else
          return accumulate(range.size().begin(), range.size().begin() + right_inner_dim());
      }

      template <typename LeftRange, typename RightRange>
      DynamicRange result_range(const LeftRange& left, const RightRange& right) const {
        // Check that the order and dimensions of the left and right tensors are correct.
        TA_ASSERT(left.order() == right.order());
        TA_ASSERT(left.dim() == left_dim());
        TA_ASSERT(right.dim() == right_dim());

        return DynamicRange(result_index(left.start(), right.start(), left.order()),
            result_index(left.finish(), right.finish(), left.order()), left.order());
      }

      /// Tensor contraction

      /// This function constructs the result tensor and calls
      /// \c contract_tensor(res,left,right) . The outer indices of \c left and
      /// \c right must be to the left.  If the data is not in the correct
      /// layout, then use the \c permute_contract_tensor() function instead.
      /// \tparam Left The left-hand-side tensor argument type
      /// \tparam Right The right-hand-side tensor argument type
      /// \param left The left-hand-side tensor argument
      /// \param right The right-hand-side tensor argument
      /// \throw TiledArray::Exception When the orders of left and right are not
      /// equal.
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c left tensor is not equal to \c left_dim() .
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c right tensor is not equal to \c right_dim() .
      /// \throw TiledArray::Exception When the inner dimensions of \c A and
      /// \c B are not coformal (i.e. the number and range of the inner
      /// dimensions are not the same).
      template <typename Left, typename Right>
      expressions::Tensor<typename ContractionValue<typename Left::value_type,
          typename Right::value_type>::type, DynamicRange>
      contract_tensor(const Left& left, const Right& right) const {

        // This function fuses the inner and outer dimensions of the left- and
        // right-hand tensors such that the contraction can be performed with a
        // matrix-matrix multiplication.

        // Construct the result tensor
        typedef typename ContractionValue<typename Left::value_type,
            typename Right::value_type>::type value_type;
        expressions::Tensor<value_type, DynamicRange>
            res(result_range(left.range(), right.range()));

        contract_tensor(res, left, right);

        return res;
      }

      /// Tensor contraction

      /// The will contract \c A with \c B and place the result in \c C.
      /// The contraction algorithms are: \n
      /// \c order=TiledArray::detail::decreasing_dimension_order
      /// \f[
      /// C_{m_1, m_2, \dots , n_1, n_2, \dots} =
      ///     \sum_{i_1, i_2, \dots}
      ///         A_{m_1, m_2, \dots, i_1, i_2, \dots}
      ///         B_{n_1, n_2, \dots, i_1, i_2, \dots}
      /// \f]
      /// \c order=TiledArray::detail::increasing_dimension_order
      /// \f[
      /// C_{m_1, m_2, \dots , n_1, n_2, \dots} =
      ///     \sum_{i_1, i_2, \dots}
      ///         A_{i_1, i_2, \dots, m_1, m_2, \dots}
      ///         B_{i_1, i_2, \dots, n_1, n_2, \dots}
      /// \f]
      /// If the data is not in the correct layout, then use the
      /// \c permute_contract_tensor() function instead.
      /// \tparam Res The result tensor type
      /// \tparam Left The left-hand-side tensor argument type
      /// \tparam Right The right-hand-side tensor argument type
      /// \param left The left-hand-side tensor argument
      /// \param right The right-hand-side tensor argument
      /// \throw TiledArray::Exception When the orders of left and right are not
      /// equal.
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c left tensor is not equal to \c left_dim() .
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c right tensor is not equal to \c right_dim() .
      /// \throw TiledArray::Exception When the inner dimensions of \c A and
      /// \c B are not coformal (i.e. the number and range of the inner
      /// dimensions are not the same).
      template <typename Res, typename Left, typename Right>
      void contract_tensor(Res& res, const Left& left, const Right& right) const {
        // Check that the order and dimensions of the left and right tensors are correct.
        TA_ASSERT(res.range().order() == left.range().order());
        TA_ASSERT(res.range().order() == right.range().order());
        TA_ASSERT(left.range().dim() == left_dim());
        TA_ASSERT(right.range().dim() == right_dim());
        TA_ASSERT(res.range().dim() == res_dim());
        TA_ASSERT(check_coformal(left.range(), right.range()));
        TA_ASSERT(check_left_coformal(res.range(), left.range()));
        TA_ASSERT(check_right_coformal(res.range(), right.range()));

        // This function fuses the inner and outer dimensions of the left- and
        // right-hand tensors such that the contraction can be performed with a
        // matrix-matrix multiplication.

        // Construct the result tensor
        typedef typename ContractionValue<typename Left::value_type,
            typename Right::value_type>::type value_type;

        const std::size_t m = left_outer(left.range());
        const std::size_t i = left_inner(left.range());
        const std::size_t n = right_outer(right.range());

        TiledArray::detail::gemm(CblasTrans, CblasNoTrans, m, n, i, value_type(1),
            left.data(), i, right.data(), i, value_type(1), res.data(), m);
      }

      /// Tensor contraction

      /// The will contract \c a with \c b , contract \c c with
      /// \c d , and return the result tensor.
      /// The contraction algorithms are: \n
      /// \c order=TiledArray::detail::decreasing_dimension_order
      /// \f[
      /// E_{m_1, m_2, \dots , n_1, n_2, \dots} =
      ///     \sum_{i_1, i_2, \dots}
      ///         A_{m_1, m_2, \dots, i_1, i_2, \dots}
      ///         B_{n_1, n_2, \dots, i_1, i_2, \dots}
      ///   + \sum_{j_1, j_2, \dots}
      ///         C_{m_1, m_2, \dots, j_1, j_2, \dots}
      ///         D_{n_1, n_2, \dots, j_1, j_2, \dots}
      /// \f]
      /// \c order=TiledArray::detail::increasing_dimension_order
      /// \f[
      /// E_{m_1, m_2, \dots , n_1, n_2, \dots} =
      ///     \sum_{i_1, i_2, \dots}
      ///         A_{i_1, i_2, \dots, m_1, m_2, \dots}
      ///         B_{i_1, i_2, \dots, n_1, n_2, \dots}
      ///    + \sum_{j_1, j_2, \dots}
      ///         C_{j_1, j_2, \dots, m_1, m_2, \dots}
      ///         D_{j_1, j_2, \dots, n_1, n_2, \dots}
      /// \f]
      /// where E is the result tensor.
      /// If the data is not in the correct layout, then use the
      /// \c permute_contract_tensor() function instead.
      /// \tparam A The left-hand tensor argument type for the first contraction
      /// \tparam B The right-hand tensor argument type for the first contraction
      /// \tparam C The left-hand tensor argument type for the second contraction
      /// \tparam D The right-hand tensor argument type for the second contraction
      /// \param a The left-hand tensor argument for the first contraction
      /// \param b The right-hand tensor argument for the first contraction
      /// \param c The left-hand tensor argument for the second contraction
      /// \param d The right-hand tensor argument for the second contraction
      /// \throw TiledArray::Exception When the orders of \c a , \c b , \c c ,
      /// and \c d are not equal.
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c a or \c c tensor is not equal to \c left_dim() .
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c b or \c d tensor is not equal to \c right_dim() .
      /// \throw TiledArray::Exception When the inner dimensions of \c a and
      /// \c b are not coformal (i.e. the number and range of the inner
      /// dimensions are not the same).
      /// \throw TiledArray::Exception When the inner dimensions of \c c and
      /// \c d are not coformal (i.e. the number and range of the inner
      /// dimensions are not the same).
      template <typename A, typename B, typename C, typename D>
      expressions::Tensor<typename ContractionValue<typename ContractionValue<typename A::value_type,
          typename B::value_type>::type, typename ContractionValue<typename C::value_type,
          typename D::value_type>::type>::type, DynamicRange>
      contract_tensor(const A& a, const B& b, const C& c, const D& d) const {
        // Check that the order and dimensions of the left and right tensors are correct.
        TA_ASSERT(a.range().order() == b.range().order());
        TA_ASSERT(c.range().order() == d.range().order());
        TA_ASSERT(a.range().order() == c.range().order());
        TA_ASSERT(a.range().dim() == left_dim());
        TA_ASSERT(b.range().dim() == right_dim());
        TA_ASSERT(c.range().dim() == left_dim());
        TA_ASSERT(d.range().dim() == right_dim());
        TA_ASSERT(check_coformal(a.range(), b.range()));
        TA_ASSERT(check_coformal(c.range(), d.range()));

        // This function fuses the inner and outer dimensions of the left- and
        // right-hand tensors such that the contraction can be performed with a
        // matrix-matrix multiplication.

        // Construct the result tensor
        typedef typename ContractionValue<
              typename ContractionValue<typename A::value_type, typename B::value_type>::type,
              typename ContractionValue<typename C::value_type, typename D::value_type>::type
            >::type value_type;
        expressions::Tensor<value_type, DynamicRange>
            res(result_range(a.range(), b.range()));

        // This function fuses the inner and outer dimensions of the left- and
        // right-hand tensors such that the contraction can be performed with a
        // matrix-matrix multiplication.

        const std::size_t m = left_outer(a.range());
        const std::size_t i = left_inner(a.range());
        const std::size_t j = left_inner(c.range());
        const std::size_t n = right_outer(b.range());

        TiledArray::detail::gemm(CblasTrans, CblasNoTrans, m, n, i, value_type(1),
            a.data(), i, b.data(), i, value_type(1), res.data(), m);
        TiledArray::detail::gemm(CblasTrans, CblasNoTrans, m, n, j, value_type(1),
            c.data(), j, d.data(), j, value_type(1), res.data(), m);

        return res;
      }

      unsigned int dim() const { return left_outer_.size() + right_outer_.size(); }
      std::size_t left_inner_dim() const { return left_inner_.size(); }
      std::size_t left_outer_dim() const { return left_outer_.size(); }
      std::size_t right_inner_dim() const { return right_inner_.size(); }
      std::size_t right_outer_dim() const { return right_outer_.size(); }
      std::size_t left_dim() const { return left_inner_dim() + left_outer_dim(); }
      std::size_t right_dim() const { return right_inner_dim() + right_outer_dim(); }
      std::size_t res_dim() const { return left_outer_dim() + right_outer_dim(); }

      /// Get the left argument variable list

      /// \return The permuted left variable list
      expressions::VariableList left_vars(const expressions::VariableList& v) const {
        if(do_perm_left_)
          return perm_left_ ^ v;

        return v;
      }

      /// Get the right argument variable list

      /// \return The permuted right variable list
      expressions::VariableList right_vars(const expressions::VariableList& v) const {
        if(do_perm_right_)
          return perm_right_ ^ v;

        return v;
      }

    private:

      /// Check that the left and right arrays are coformal

      ///
      template <typename LeftRange, typename RightRange>
      bool check_coformal(const LeftRange& left, const RightRange& right) const {
        if(left.order() == TiledArray::detail::decreasing_dimension_order) {
          return std::equal(left.start().begin() + left_outer_dim(), left.start().end(),
              right.start().begin() + right_outer_dim())
              && std::equal(left.finish().begin() + left_outer_dim(), left.finish().end(),
              right.finish().begin() + right_outer_dim());
        } else {
          return std::equal(left.start().begin(), left.start().begin() + left_inner_dim(),
              right.start().begin())
              && std::equal(left.finish().begin(), left.finish().begin() + left_inner_dim(),
              right.finish().begin());
        }
      }

      /// Check that the result and left arrays are coformal
      template <typename ResRange, typename LeftRange>
      bool check_left_coformal(const ResRange& res, const LeftRange& left) const {
        const TiledArray::detail::DimensionOrderType order = left.order();
        return std::equal(res.start().begin(), res.start().begin() + left_outer_dim(),
            (order == TiledArray::detail::decreasing_dimension_order ? left.start().begin() : left.start().begin() + left_inner_dim()))
            && std::equal(res.finish().begin(), res.finish().begin() + left_outer_dim(),
            (order == TiledArray::detail::decreasing_dimension_order ? left.finish().begin() : left.finish().begin() + left_inner_dim()));
      }

      /// Check that the result and right arrays are coformal
      template <typename ResRange, typename RightRange>
      bool check_right_coformal(const ResRange& res, const RightRange& right) const {
        const TiledArray::detail::DimensionOrderType order = right.order();
        return std::equal(res.start().begin() + left_outer_dim(), res.start().end(),
            (order == TiledArray::detail::decreasing_dimension_order ? right.start().begin() : right.start().begin() + right_inner_dim()))
            && std::equal(res.finish().begin() + left_outer_dim(), res.finish().end(),
            (order == TiledArray::detail::decreasing_dimension_order ? right.finish().begin() : right.finish().begin() + right_inner_dim()));
      }

      /// Product accumulation

      ///
      /// \tparam InIter The input iterator type
      /// \param first The start of the iterator range to be accumulated
      /// \param first The end of the iterator range to be accumulated
      /// \return The product of each value in the iterator range.
      template <typename InIter>
      static typename std::iterator_traits<InIter>::value_type accumulate(InIter first, InIter last) {
        typename std::iterator_traits<InIter>::value_type result = 1ul;
        for(; first != last; ++first)
          result *= *first;
        return result;
      }

      template <typename LeftIndex, typename RightIndex>
      DynamicRange::index result_index(const LeftIndex& l_index,
          const RightIndex& r_index, const detail::DimensionOrderType& order) const {
        DynamicRange::index result(dim());

        if(order == detail::decreasing_dimension_order)
          std::copy(r_index.begin(), r_index.begin() + right_outer_dim(),
              std::copy(l_index.begin(), l_index.begin() + left_outer_dim(), result.begin()));
        else
          std::copy(r_index.begin() + right_inner_dim(), r_index.end(),
              std::copy(l_index.begin() + left_inner_dim(), l_index.end(), result.begin()));

        return result;
      }


      static void init_inner_outer(const expressions::VariableList& first,
          const expressions::VariableList& second, map_type& inner, map_type& outer)
      {
        // Reserve storage for maps.
        inner.reserve(first.dim());
        outer.reserve(first.dim());

        // construct the inner and outer maps.
        for(expressions::VariableList::const_iterator it = first.begin(); it != first.end(); ++it) {
          expressions::VariableList::const_iterator second_it =
              std::find(second.begin(), second.end(), *it);
          if(second_it == second.end())
            outer.push_back(std::distance(first.begin(), it));
          else
            inner.push_back(std::distance(first.begin(), it));
        }
      }

      static void init_permutation(const expressions::VariableList& in_vars,
          const map_type& inner, const map_type& outer, const detail::DimensionOrderType& order,
          Permutation& perm, bool& do_perm)
      {
        std::vector<std::string> vars;
        vars.reserve(in_vars.size());

        if(order == TiledArray::detail::decreasing_dimension_order) {
          for(map_type::const_iterator it = outer.begin(); it != outer.end(); ++it)
            vars.push_back(in_vars[*it]);
          for(map_type::const_iterator it = inner.begin(); it != inner.end(); ++it)
            vars.push_back(in_vars[*it]);
        } else {
          for(map_type::const_iterator it = inner.begin(); it != inner.end(); ++it)
            vars.push_back(in_vars[*it]);
          for(map_type::const_iterator it = outer.begin(); it != outer.end(); ++it)
            vars.push_back(in_vars[*it]);
        }
        expressions::VariableList perm_vars(vars.begin(), vars.end());
        perm = perm_vars.permutation(in_vars);

        do_perm = perm_vars != in_vars;
      }


      map_type left_inner_;
      map_type left_outer_;
      map_type right_inner_;
      map_type right_outer_;

      Permutation perm_left_;
      Permutation perm_right_;

      bool do_perm_left_;
      bool do_perm_right_;
    }; // class Contraction

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_H__INCLUDED
