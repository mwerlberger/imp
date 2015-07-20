#ifndef IMP_CU_CONST_MATRIX_CUH
#define IMP_CU_CONST_MATRIX_CUH

#include <ostream>
#include <cuda_runtime.h>
#include <imp/core/pixel.hpp>
#include <imp/cu_core/cu_memory_storage.cuh> // Deallocator
#include <memory> // std::unique_ptr
#include <Eigen/Dense>
#include <kindr/minimal/quat-transformation.h>

namespace imp{
namespace cu{

// This class is used for __shfl operations. Therefore sizeof(ConstMatrix) has to be a multiple of 4.
// Furthermore, the default constructor has to be used.

template<typename _Type, size_t _rows, size_t _cols>
class ConstMatrixMemoryHandler
{
public:
  using Type = _Type;
  using Memory = imp::cu::MemoryStorage<Type>;

  enum class MemoryType {HOST_MEMORY,DEVICE_MEMORY};

  ConstMatrixMemoryHandler(Type* src, MemoryType mem_type_ = MemoryType::DEVICE_MEMORY)
    :mem_type_(mem_type_)
  {
    setupMemory(src);
  }

  ConstMatrixMemoryHandler(Eigen::Matrix<Type,_rows,_cols,Eigen::RowMajor> mat_row_major,
                           MemoryType mem_type_ = MemoryType::DEVICE_MEMORY)
    :mem_type_(mem_type_)
  {
    setupMemory(mat_row_major.data());
  }

//  ConstMatrixMemoryHandler(Eigen::Matrix<float,_rows,_cols,Eigen::ColMajor> mat_col_major,
//                           MemoryType mem_type_ = MemoryType::DEVICE_MEMORY)
//    :mem_type_(mem_type_)
//  {
//    Eigen::Matrix<float,_rows,_cols,Eigen::RowMajor> mat_col_major.transpose();
//    setupMemory(mat_row_major.data());
//  }

  ConstMatrixMemoryHandler(kindr::minimal::QuatTransformationTemplate<Type> T_quat,
                           MemoryType mem_type_ = MemoryType::DEVICE_MEMORY)
    :mem_type_(mem_type_)
  {
    if(_rows!=3||_cols!=4)
      IMP_THROW_EXCEPTION("matrix  has to be of size 3x4 to hold a transformation matrix");
    Eigen::Matrix<Type,3,4,Eigen::RowMajor> T_eig = T_quat.getTransformationMatrix();
    setupMemory(T_eig.data());
  }

  ~ConstMatrixMemoryHandler()
  {
    if(mem_type_ == MemoryType::HOST_MEMORY)
    {
      free(data_);
      std::cout << "free memory CPU" << std::endl;
    }
    else if(mem_type_ == MemoryType::DEVICE_MEMORY)
    {
      cudaFree(data_);
      std::cout << "free memory GPU" << std::endl;
    }
  }

  inline Type* get() const { return data_;}

private:
  void setupMemory(const Type* src)
  {
    if(mem_type_ == MemoryType::HOST_MEMORY)
    {
      data_ = (Type*) malloc (sizeof(Type)*_rows*_cols);
      if(data_ == NULL)
      {
        throw std::bad_alloc();;
      }
      memcpy(data_,src,_rows*_cols*sizeof(Type));
      std::cout << "copy memory CPU" << std::endl;
    }
    else if(mem_type_ == MemoryType::DEVICE_MEMORY)
    {
      data_ = Memory::alloc(_rows*_cols);
      const cudaError cu_err =
          cudaMemcpy(data_,src,_rows*_cols*sizeof(Type)
                     ,cudaMemcpyHostToDevice);

      if (cu_err != cudaSuccess)
        IMP_CU_THROW_EXCEPTION("cudaMemcpy returned error code", cu_err);
      std::cout << "copy memory GPU" << std::endl;
    }
  }

  size_t rows_ = _rows;
  size_t cols_ = _cols;
  MemoryType mem_type_;
  Type* data_;
};

template<typename _Type, size_t _rows, size_t _cols>
class ConstMatrixDynamic
{
  using Type = _Type;

public:
  __host__
  ConstMatrixDynamic() { }

  __host__
  ~ConstMatrixDynamic() { }

  __host__
  ConstMatrixDynamic(ConstMatrixMemoryHandler<Type,_rows,_cols>& mem_handler)
  {
    data_ = mem_handler.get();
  }

  __host__ __device__ __forceinline__
  size_t rows() const { return rows_; }

  __host__ __device__ __forceinline__
  size_t cols() const { return cols_; }

  /** Data access operator given a \a row and a \a col
   * @return unchangable value at \a (row,col)
   */
//  __device__ __forceinline__
//  const Type& ldg(int row, int col) const
//  {
//    return __ldg(&data_[col*rows_ + row]);
//  }

  __device__ __forceinline__
  const Type& operator()(int row, int col) const
  {
    //return __ldg(&data_[row*cols_ + col]);
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

protected:
protected:
  size_t rows_ = _rows;
  size_t cols_ = _cols;
  Type* data_;
};

template<typename _Type, size_t _rows, size_t _cols>
class ConstMatrixStatic
{
  using Type = _Type;

public:
  __host__
  ConstMatrixStatic() { }

  __host__
  ~ConstMatrixStatic() { }

  __host__
  ConstMatrixStatic(Eigen::Matrix<Type,_rows,_cols,Eigen::RowMajor>& input_row_major)
  {
    for(int ii=0; ii < _rows*_cols;ii++)
    {
      data_[ii] = input_row_major(ii);
    }
  }

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

  /** Data access operator given an \a index
   * @return unchangable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  const Type& operator[](int ind) const
  {
    return data_[ind];
  }

protected:
protected:
  size_t rows_ = _rows;
  size_t cols_ = _cols;
  Type data_[_rows*_cols];
};

//matrix vector multiplication
__host__ __device__ __forceinline__
float3 transform(const ConstMatrixStatic<float,3,4>& T, const float3& v)
{
  return make_float3(
        T(0,0)*v.x + T(0,1)*v.y + T(0,2)*v.z + T(0,3),
        T(1,0)*v.x + T(1,1)*v.y + T(1,2)*v.z + T(1,3),
        T(2,0)*v.x + T(2,1)*v.y + T(2,2)*v.z + T(2,3)
        );
}

__device__  __forceinline__
float3 transform(const ConstMatrixDynamic<float,3,4>& T, const float3& v)
{
  return make_float3(
        T(0,0)*v.x + T(0,1)*v.y + T(0,2)*v.z + T(0,3),
        T(1,0)*v.x + T(1,1)*v.y + T(1,2)*v.z + T(1,3),
        T(2,0)*v.x + T(2,1)*v.y + T(2,2)*v.z + T(2,3)
        );
}

//__device__  __forceinline__
//float3 transformLdg(const ConstMatrixDynamic<float,3,4>& T, const float3& v)
//{
//  return make_float3(
//        T.ldg(0,0)*v.x + T.ldg(0,1)*v.y + T.ldg(0,2)*v.z + T.ldg(0,3),
//        T.ldg(1,0)*v.x + T.ldg(1,1)*v.y + T.ldg(1,2)*v.z + T.ldg(1,3),
//        T.ldg(2,0)*v.x + T.ldg(2,1)*v.y + T.ldg(2,2)*v.z + T.ldg(2,3)
//        );
//}


// convenient typedefs
typedef ConstMatrixMemoryHandler<float,3,4> TransformationMemoryHdlr;
typedef ConstMatrixDynamic<float,3,4> TransformationDynamic;
typedef ConstMatrixStatic<float,3,4> TransformationStatic;

#if 0
template<typename _Type, size_t _rows, size_t _cols>
class ConstMatrixStatic: public ConstMatrixBase<_Type,_rows,_cols,ConstMatrixStatic>
{
  using Type = _Type;

public:
  __host__
  ConstMatrixStatic() { }

  __host__
  ~ConstMatrixStatic() { }

  /** Data access operator given a \a row and a \a col
   * @return unchangable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  const Type& operator()(int row, int col) const
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

protected:
  Type data_[_rows*_cols];
};


template<typename _Type, size_t _rows, size_t _cols>
class ConstMatrix
{
  using Type = _Type;
  using Deallocator = imp::cu::MemoryDeallocator<Type>;

public:
  __host__
  ConstMatrix() { }

  __host__
  ~ConstMatrix() { }

  //  __host__ __device__
  //  ConstMatrix(const ConstMatrix& other)
  //  {
  //    // TODO
  //  }

  //  __host__ __device__
  //  ConstMatrix& operator=(const ConstMatrix& other)
  //  {
  //    // TODO
  //    return this;
  //  }

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

  /** Data access operator given an \a index
   * @return unchangable value at \a (row,col)
   */
  __host__ __device__ __forceinline__
  const Type& operator[](int ind) const
  {
    return data_[ind];
  }

protected:
  std::unique_ptr<Type,Deallocator> data_;
  size_t rows_ = _rows;
  size_t cols_ = _cols;
};


template<typename Type>
class ConstMatrix3X4: public ConstMatrix<Type,3,4>
{
  using Base = ConstMatrix<Type,3,4>;
  using Base::data_;
  using Memory = imp::cu::MemoryStorage<Type>;

private:
  static constexpr int kSizeMatrix3x4 = 12;
public:
  ConstMatrix3X4() { }
  ~ConstMatrix3X4() { }

  ConstMatrix3X4(Type* transformation_row_maj_3x4)
  {
    data_.reset(Memory::alloc(kSizeMatrix3x4));
    const cudaError cu_err =
        cudaMemcpy(data_.get(),transformation_row_maj_3x4,kSizeMatrix3x4*sizeof(Type)
                   ,cudaMemcpyHostToDevice);

    if (cu_err != cudaSuccess)
      IMP_CU_THROW_EXCEPTION("cudaMemcpy returned error code", cu_err);
  }

  ConstMatrix3X4(Eigen::Matrix<Type,3,3> rot, Eigen::Matrix<Type,3,1> trans)
  {
    Eigen::Matrix<Type,3,4> T;
    T.block<3,3>(0,0) = rot;
    T.col(3) = trans;

    data_.reset(Memory::alloc(kSizeMatrix3x4));
    const cudaError cu_err =
        cudaMemcpy(data_.get(),T.data(),kSizeMatrix3x4*sizeof(Type)
                   ,cudaMemcpyHostToDevice);

    if (cu_err != cudaSuccess)
      IMP_CU_THROW_EXCEPTION("cudaMemcpy returned error code", cu_err);
  }

  ConstMatrix3X4(Eigen::Matrix<Type,3,4> T)
  {
    data_.reset(Memory::alloc(kSizeMatrix3x4));
    const cudaError cu_err =
        cudaMemcpy(data_.get(),T.data(),kSizeMatrix3x4*sizeof(Type)
                   ,cudaMemcpyHostToDevice);

    if (cu_err != cudaSuccess)
      IMP_CU_THROW_EXCEPTION("cudaMemcpy returned error code", cu_err);
  }

  ConstMatrix3X4(kindr::minimal::QuatTransformationTemplate<Type> T_quat)
  {
    Eigen::Matrix<Type,3,4> T_eig = T_quat.getTransformationMatrix();
    data_.reset(Memory::alloc(T_eig.data()));
    const cudaError cu_err =
        cudaMemcpy(data_.get(),T_eig.data(),kSizeMatrix3x4*sizeof(Type)
                   ,cudaMemcpyHostToDevice);

    if (cu_err != cudaSuccess)
      IMP_CU_THROW_EXCEPTION("cudaMemcpy returned error code", cu_err);
  }


  __host__
  void resetData(Type* transformation_row_maj_3x4)
  {
    data_.reset(Memory::alloc(kSizeMatrix3x4));
    const cudaError cu_err =
        cudaMemcpy(data_.get(),transformation_row_maj_3x4,kSizeMatrix3x4*sizeof(Type)
                   ,cudaMemcpyHostToDevice);

    if (cu_err != cudaSuccess)
      IMP_CU_THROW_EXCEPTION("cudaMemcpy returned error code", cu_err);
  }
};


// convenient typedefs
typedef ConstMatrix3X4<float> Transformation;
typedef ConstMatrix3X4<double> TransformationDouble;
#endif

//matrix vector multiplication
//__device__ __forceinline__
//float3 transform(const Transformation& T, const float3& v)
//{
//  return make_float3(
//        T(0,0)*v.x + T(0,1)*v.y + T(0,2)*v.z + T(0,3),
//        T(1,0)*v.x + T(1,1)*v.y + T(1,2)*v.z + T(1,3),
//        T(2,0)*v.x + T(2,1)*v.y + T(2,2)*v.z + T(2,3)
//        );
//}

//double3 transform(const TransformationDouble& T, const double3& v)
//{
//  return make_double3(
//        T(0,0)*v.x + T(0,1)*v.y + T(0,2)*v.z + T(0,3),
//        T(1,0)*v.x + T(1,1)*v.y + T(1,2)*v.z + T(1,3),
//        T(2,0)*v.x + T(2,1)*v.y + T(2,2)*v.z + T(2,3)
//        );
//}

#if 0

//template<typename Type>
//__host__
//void init(Type* rotation)

//------------------------------------------------------------------------------
// convenience typedefs
using Matrix3f = Transformation<float>;
using Vector3f = ConstMatrix<float,1,3>;

//==============================================================================


//------------------------------------------------------------------------------
template<typename Type, size_t _rows, size_t CR, size_t _cols>
__host__ __device__ __forceinline__
ConstMatrix<Type, _rows, _cols> operator*(const ConstMatrix<Type, _rows, CR> & lhs,
                                          const ConstMatrix<Type, CR, _cols> & rhs)
{
  ConstMatrix<Type, _rows, _cols> result;
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
ConstMatrix<Type, 2, 2> invert(const ConstMatrix<Type, 2, 2> & in)
{
  ConstMatrix<Type, 2, 2> out;
  float det = in[0]*in[3] - in[1]*in[2];
  out[0] =  in[3] / det;
  out[1] = -in[1] / det;
  out[2] = -in[2] / det;
  out[3] =  in[0] / det;
  return out;
}


//------------------------------------------------------------------------------
// matrix vector multiplication
__host__ __device__ __forceinline__
float3 operator*(const Matrix3f& mat, const float3& v)
{
  return make_float3(
        mat(0,0)*v.x + mat(0,1)*v.y + mat(0,2)*v.z,
        mat(1,0)*v.x + mat(1,1)*v.y + mat(1,2)*v.z,
        mat(2,0)*v.x + mat(2,1)*v.y + mat(2,2)*v.z
        );
}

//------------------------------------------------------------------------------
// matrix vector multiplication
__host__ __device__ __forceinline__
Vec32fC3 operator*(const Matrix3f& mat, const Vec32fC3& v)
{
  return Vec32fC3(
        mat(0,0)*v.x + mat(0,1)*v.y + mat(0,2)*v.z,
        mat(1,0)*v.x + mat(1,1)*v.y + mat(1,2)*v.z,
        mat(2,0)*v.x + mat(2,1)*v.y + mat(2,2)*v.z
        );
}

//------------------------------------------------------------------------------
template<typename T, size_t rows, size_t cols>
__host__
inline std::ostream& operator<<(std::ostream &os,
                                const cu::ConstMatrix<T, rows, cols>& m)
{
  os << "[";
  for (int r=0; r<rows; ++r)
  {
    for (int c=0; c<cols; ++c)
    {
      os << m(r,c);
      if (c<cols-1)
        os << ",";
    }
    os << "; ";
  }
  os << "]";
  return os;
}
#endif

}
}

#endif // IMP_CU_CONST_MATRIX_CUH

