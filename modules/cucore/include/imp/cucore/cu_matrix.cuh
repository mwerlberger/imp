#ifndef IMP_CU_MATRIX_CUH
#define IMP_CU_MATRIX_CUH

#include <cuda_runtime.h>
#include <array>

namespace imp{
namespace cu{

//------------------------------------------------------------------------------
template<typename _Type, size_t _rows, size_t _cols>
class Matrix
{
  using Type = _Type;

public:
  __host__ __device__
  Matrix() = default;

  __host__ __device__
  ~Matrix() = default;

  __host__ __device__ __forceinline__
  size_t rows() const { return rows_; }

  __host__ __device__ __forceinline__
  size_t cols() const { return cols_; }

  /** Data access operator given a \a row and a \a col
   * @return unchangable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  const Type& operator()(int row, int col) const
  {
    return data_[row*cols_ + col];
  }

  /** Data access operator given a \a row and a \a col
   * @return changable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  Type& operator()(int row, int col)
  {
    return data_[row*cols_ + col];
  }

  /** Data access operator given an \a index
   * @return unchangable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  const Type& operator[](int ind) const
  {
    return data_[ind];
  }

  /** Data access operator given an \a index
   * @return changable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  Type & operator[](int ind)
  {
    return data_[ind];
  }

#if 0
  template<typename TypeFrom>
  __host__ inline Matrix(const Eigen::Matrix<TypeFrom,R,C>& mat)
  {
    for (size_t row=0; row<R; ++row)
    {
      for (size_t col=0; col<C; ++col)
      {
        data[row*C+col] = (Type)mat(row,col);
      }
    }
  }
#endif

private:
  Type data_[_rows*_cols];
  size_t rows_ = _rows;
  size_t cols_ = _cols;
};

//------------------------------------------------------------------------------
// convenience typedefs
using Matrix3f = Matrix<float,3,3>;
using Vector3f = Matrix<float,1,3>;

//==============================================================================


//------------------------------------------------------------------------------
template<typename Type, size_t _rows, size_t CR, size_t _cols>
__host__ __device__ __forceinline__
Matrix<Type, _rows, _cols> operator*(const Matrix<Type, _rows, CR> & lhs,
                                     const Matrix<Type, CR, _cols> & rhs)
{
  Matrix<Type, _rows, _cols> result;
  for(size_t row=0; row<_rows; ++row)
  {
    for(size_t col=0; col<_cols; ++col)
    {
      result(row, col) = 0;
      for(size_t i=0; i<CR; ++i)
      {
        result(row, col) += lhs(row,i) * rhs(i,col);
      }
    }
  }
  return result;
}

//------------------------------------------------------------------------------
template<typename Type>
__host__ __device__ __forceinline__
Matrix<Type, 2, 2> invert(const Matrix<Type, 2, 2> & in)
{
  Matrix<Type, 2, 2> out;
  float det = in[0]*in[3] - in[1]*in[2];
  out[0] =  in[3] / det;
  out[1] = -in[1] / det;
  out[2] = -in[2] / det;
  out[3] =  in[0] / det;
  return out;
}


//------------------------------------------------------------------------------
// matrix vector multiplication
__host__ __device__ inline
float3 operator*(const Matrix3f& mat, const float3& v)
{
  return make_float3(
        mat(0,0)*v.x + mat(0,1)*v.y + mat(0,2)*v.z,
        mat(1,0)*v.x + mat(1,1)*v.y + mat(1,2)*v.z,
        mat(2,0)*v.x + mat(2,1)*v.y + mat(2,2)*v.z
        );
}


}
}

#endif // IMP_CU_MATRIX_CUH

