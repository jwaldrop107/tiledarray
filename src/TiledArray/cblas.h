#ifndef TILEDARRAY_CBLAS_H__INCLUDED
#define TILEDARRAY_CBLAS_H__INCLUDED

#include <TiledArray/config.h>
#include <Eigen/Core>

#if defined(TILEDARRAY_HAS_CBLAS) && defined(HAVE_MKL_H)
#include <mkl.h>
#elif defined(TILEDARRAY_HAS_CBLAS) && defined(HAVE_CBLAS_H)
extern "C" {
#include <cblas.h>
}
#elif defined(TILEDARRAY_HAS_BLAS)

extern "C" void dgemm(const char *opa, const char *opb, const long *m, const long *n, const long *k,
                       const real8 *alpha, const real8 *a, const long *lda, const real8 *b, const long *ldb,
                       const real8 *beta, real8 *c, const long *ldc, char_len opalen, char_len opblen);

extern "C" void sgemm(const char *opa, const char *opb, const long *m, const long *n, const long *k,
                       const real4 *alpha, const real4 *a, const long *lda, const real4 *b, const long *ldb,
                       const real4 *beta, real4 *c, const long *ldc, char_len opalen, char_len opblen);

extern "C" void zgemm(const char *opa, const char *opb, const long *m, const long *n, const long *k,
                       const complex_real8 *alpha,
                       const complex_real8 *a, const long *lda, const complex_real8 *b, const long *ldb,
                       const complex_real8 *beta, complex_real8 *c, const long *ldc,  char_len opalen, char_len opblen);

extern "C" void cgemm(const char *opa, const char *opb, const long *m, const long *n, const long *k,
                       const complex_real4 *alpha,
                       const complex_real4 *a, const long *lda, const complex_real4 *b, const long *ldb,
                       const complex_real4 *beta, complex_real4 *c, const long *ldc, char_len opalen, char_len opblen);
#endif

#ifndef TILEDARRAY_HAS_CBLAS

enum CBLAS_TRANSPOSE {CblasNoTrans=0, CblasTrans=1, CblasConjTrans=2};

#endif // TILEDARRAY_HAS_CBLAS

namespace TiledArray {
  namespace detail {

    template <typename T>
    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const T alpha, const T* a, const long lda, const T*b, const long ldb,
        const T beta, T* c, const long ldc)
    {
      // The ConjTrans operation is not implemented because it is not used.
      TA_ASSERT(transa != CblasConjTrans);
      TA_ASSERT(transb != CblasConjTrans);

      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_type;
      typedef Eigen::Map<matrix_type, Eigen::AutoAlign, Eigen::OuterStride<> > map_type;
      typedef Eigen::Map<const matrix_type, Eigen::AutoAlign, Eigen::OuterStride<> > const_map_type;

      const_map_type A(a, (transa  == CblasNoTrans ? m : k), (transa  == CblasNoTrans ? k : m), Eigen::OuterStride<>(lda));
      const_map_type B(b, (transb  == CblasNoTrans ? k : n), (transb  == CblasNoTrans ? n : k), Eigen::OuterStride<>(ldb));
      map_type C(c, m, n, Eigen::OuterStride<>(ldc));

      C *= beta;
      if(transa == CblasNoTrans) {
        if(transb == CblasNoTrans) {
          C.noalias() += alpha * A * B;
        } else {
          C.noalias() += alpha * A * B.transpose();
        }
      } else {
        if(transb == CblasNoTrans) {
          C.noalias() += alpha * A.transpose() * B;
        } else {
          C.noalias() += alpha * A.transpose() * B.transpose();
        }
      }
    }


#if defined(TILEDARRAY_HAS_CBLAS) || defined(HAVE_MKL_H)

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const double alpha, const double* a,
        const long lda, const double* b, const long ldb,
        const double beta, double* c, const long ldc)
    {
      cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const std::complex<double> alpha, const std::complex<double>* a,
        const long lda, const std::complex<double>* b, const long ldb,
        const std::complex<double> beta, std::complex<double>* c, const long ldc)
    {
      cblas_zgemm(CblasColMajor, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const float alpha, const float* a, const long lda, const float* b,
        const long ldb, const float beta, float* c, const long ldc)
    {
      cblas_sgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const std::complex<float> alpha, const std::complex<float>* a,
        const long lda, const std::complex<float>* b, const long ldb,
        const std::complex<float> beta, std::complex<float>* c, const long ldc)
    {
      cblas_cgemm(CblasColMajor, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }

#elif defined(TILEDARRAY_HAS_BLAS)

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        long m, long n, long k,
        double alpha, const double* a, long lda,
        const double* b, long ldb,
        double beta, double* c, long ldc)
    {
      static const char* op[] = {"n","t","c"};
      dgemm(op[transa], op[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,
          c, &ldc, 1, 1);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        long m, long n, long k,
        std::complex<double> alpha, const std::complex<double>* a, long lda,
        const std::complex<double>* b, long ldb,
        std::complex<double> beta, std::complex<double>* c, long ldc)
    {
      static const char* op[] = {"n","t","c"};
      zgemm(op[transa], op[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,
          c, &ldc, 1, 1);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        long m, long n, long k,
        float alpha, const float* a, long lda,
        const float* b, long ldb,
        float beta, float* c, long ldc)
    {
      static const char* op[] = {"n","t","c"};
      sgemm(op[transa], op[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,
          c, &ldc, 1, 1);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        long m, long n, long k,
        std::complex<float> alpha, const std::complex<float>* a, long lda,
        const std::complex<float>* b, long ldb,
        std::complex<float> beta, std::complex<float>* c, long ldc)
    {
      static const char* op[] = {"n","t","c"};
      cgemm(op[transa], op[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,
          c, &ldc, 1, 1);
    }

#endif

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_CBLAS_H__INCLUDED
