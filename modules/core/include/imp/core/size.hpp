#ifndef IMP_SIZE_HPP
#define IMP_SIZE_HPP

#include <cstdint>
#include <array>
#include <algorithm>
#include <iostream>

namespace imp {

//------------------------------------------------------------------------------
/**
 * @brief The class SizeBase defines the templated base class of any size utilizing the CRTP pattern.
 */
template<typename T, std::uint8_t DIM, typename Derived>
struct SizeBase
{
  using size_type = std::size_t;

  std::array<T, DIM> sz; //!< internal data storage for all dimensions' sizes

  SizeBase()
  {
    std::fill(sz.begin(), sz.end(), 0);
  }

  SizeBase(const std::array<T,DIM>& arr)
    : sz(arr)
  {
  }

  virtual ~SizeBase() = default;

  SizeBase(const SizeBase& from)
    : sz(from.sz)
  {
  }

  SizeBase& operator= (const SizeBase& from)
  {
    this->sz = from.sz;
    return *this;
  }

//  SizeBase& operator* (const double factor) const
//  {
//    std::array<T,Dim> arr;
//    for(std::size_t i=0; i<sz.size(); ++i)
//    {
//      arr[i] = static_cast<T>(sz[i] * factor);
//    }
//  }

//  IuSize operator/ (const double factor) const
//  {
//    IU_ASSERT(factor != 0);
//    double invFactor = 1 / factor;
//    return IuSize(this->width, this->height, this->depth) * invFactor;
//  }


  /**
   * @brief operator [] returns the reference to the element storing the size
   *        of the \a n-the dimension
   * @param n dimension index
   * @return Reference to size element of the n-th dimension
   */
  T& operator[] (std::uint8_t n) noexcept {return sz[n];}

  /**
   * @brief operator [] returns the reference to the element storing the size
   *        of the \a n-the dimension
   * @param n dimension index
   * @return Reference to size element of the n-th dimension
   */
  constexpr const T& operator[] (std::uint8_t n) const noexcept { return sz[n]; }

  /**
   * @brief dim Returns the dimension of the size object.
   * @return Dimension.
   */
  constexpr std::uint8_t dim() const noexcept {return DIM;}


  /**
   * @brief data gives access to the underlying (raw) data storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  T* data() {return sz.data();}

  /**
   * @brief data gives access to the underlying (raw) const data storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  const T* data() const {return reinterpret_cast<const T*>(sz.data());}

  /**
   * @brief array returns a reference to the internal array used for storing the data
   */
  std::array<T, DIM>& array() {return sz;}
  /**
   * @brief array returns a const reference to the internal array used for storing the data
   */
  const std::array<T, DIM>& array() const {return reinterpret_cast<const std::array<T, DIM>&>(sz);}
};

//------------------------------------------------------------------------------
// relational operators

template<typename T, std::uint8_t DIM, typename Derived>
inline bool operator==(const SizeBase<T, DIM, Derived>& lhs,
                       const SizeBase<T, DIM, Derived>& rhs)
{
  return (lhs.array() == rhs.array());
}

template<typename T, std::uint8_t DIM, typename Derived>
inline bool operator!=(const SizeBase<T, DIM, Derived>& lhs,
                       const SizeBase<T, DIM, Derived>& rhs)
{
  return (lhs.array() != rhs.array());
}

//! @todo (MWE) the following rational comparisons cannot be used directly
//!             as they use lexicographical_compare operators

//template<typename T, std::uint8_t DIM, typename Derived>
//inline bool operator>(const SizeBase<T, DIM, Derived>& lhs,
//                      const SizeBase<T, DIM, Derived>& rhs)
//{
//  return (lhs.array() > rhs.array());
//}

//template<typename T, std::uint8_t DIM, typename Derived>
//inline bool operator>=(const SizeBase<T, DIM, Derived>& lhs,
//                      const SizeBase<T, DIM, Derived>& rhs)
//{
//  return (lhs.array() >= rhs.array());
//}

//template<typename T, std::uint8_t DIM, typename Derived>
//inline bool operator<(const SizeBase<T, DIM, Derived>& lhs,
//                      const SizeBase<T, DIM, Derived>& rhs)
//{
//  return (lhs.array() < rhs.array());
//}

//template<typename T, std::uint8_t DIM, typename Derived>
//inline bool operator<=(const SizeBase<T, DIM, Derived>& lhs,
//                      const SizeBase<T, DIM, Derived>& rhs)
//{
//  return (lhs.array() <= rhs.array());
//}


template<typename T, std::uint8_t DIM, typename Derived>
inline std::ostream& operator<<(std::ostream &os, const SizeBase<T, DIM, Derived>& rhs)
{
  auto it = rhs.array().begin();
  os << "(" << *it;
  ++it;
  for (; it != rhs.array().end(); ++it)
  {
    os << "," << *it;
  }
  os << ")";
  return os;
}


//------------------------------------------------------------------------------
/**
 * @brief The class Size defines a generic size implementation for \a DIM dimensions
 */
template<typename T, std::uint8_t DIM>
struct Size
    : public SizeBase<T, DIM, Size<T, DIM> >
{
  using Base = SizeBase<T, DIM, Size<T, DIM> >;
  using Base::Base;
  Size() = default;
  virtual ~Size() = default;
};

//------------------------------------------------------------------------------
/**
 * @brief The Size<T, 2> is a special size for a 2D shape defining its width and height
 */
template<typename T>
struct Size<T, 2>
    : public SizeBase<T, 2, Size<T, 2> >
{
  using Base = SizeBase<T, 2, Size<T, 2> >;
  using Base::Base;
  Size() = default;
  virtual ~Size() = default;

  Size(const T& width, const T& height)
    : Base({width, height})
  {
  }

  /**
   * @brief width returns the width of the 2d size
   */
  T width() const {return this->sz[0];}

  /**
   * @brief height returns the length of the second dimension of the 2d size
   * @return
   */
  T height() const {return this->sz[1];}
};

//------------------------------------------------------------------------------
/**
 * @brief The Size<T, 3> is a special size for a 3D shape defining its width, height and depth
 */
template<typename T>
struct Size<T, 3>
    : public SizeBase<T, 3, Size<T, 3> >
{
  using Base = SizeBase<T, 3, Size<T, 3> >;
  using Base::Base;
  Size() = default;
  virtual ~Size() = default;

  Size(const T& width, const T& height, const T& depth)
    : Base({width, height, depth})
  {
  }

  /**
   * @brief width returns the width of the 3d size
   */
  T width() const {return this->sz[0];}

  /**
   * @brief height returns the length of the second dimension of the 3d size
   */
  T height() const {return this->sz[1];}

  /**
   * @brief depth returns the length of the third dimension of the 3d size
   */
  T depth() const {return this->sz[2];}
};

//------------------------------------------------------------------------------
// some convencience typedefs

// 2D
typedef Size<std::uint32_t, 2> Size2u;
typedef Size<std::int32_t, 2> Size2i;
typedef Size<float, 2> Size2f;
typedef Size<float, 2> Size2d;
//3D
typedef Size<std::uint32_t, 3> Size3u;
typedef Size<std::int32_t, 3> Size3i;
typedef Size<float, 3> Size3f;
typedef Size<float, 3> Size3d;

} // namespace imp

#endif // IMP_SIZE_HPP
