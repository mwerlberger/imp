#ifndef IMP_SIZE_HPP
#define IMP_SIZE_HPP

#include <cstdint>
#include <array>
#include <algorithm>

namespace imp {

//------------------------------------------------------------------------------
/**
 * @brief The class SizeBase defines the templated base class of any size utilizing the CRTP pattern.
 */
template<typename T, std::uint8_t DIM, typename Derived>
class SizeBase
{
public:
  SizeBase()
  {
    std::fill(sz_.begin(), sz_.end(), 0);
  }

  SizeBase(const std::array<T,DIM>& arr)
    : sz_ (arr)
  {
  }

  virtual ~SizeBase() = default;

  SizeBase(const SizeBase& from)
    : sz_(from.sz_)
  {
  }

  SizeBase& operator= (const SizeBase& from)
  {
    this->sz_ = from.sz_;
  }

  /**
   * @brief dim Returns the dimension of the size object.
   * @return Dimension.
   */
  std::uint8_t dim() const {return DIM;}

  /**
   * @brief data gives access to the underlying (raw) data storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  T* data() {return sz_.data();}

  /**
   * @brief data gives access to the underlying (raw) const data storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  const T* data() const {return reinterpret_cast<const T*>(sz_.data());}

  /**
   * @brief array returns a reference to the internal array used for storing the data
   */
  std::array<T, DIM>& array() {return sz_;}
  /**
   * @brief array returns a const reference to the internal array used for storing the data
   */
  const std::array<T, DIM>& array() const {return reinterpret_cast<const std::array<T, DIM>&>(sz_);}

protected:
  std::array<T, DIM> sz_;
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


//------------------------------------------------------------------------------
/**
 * @brief The class Size defines a generic size implementation for \a DIM dimensions
 */
template<typename T, std::uint8_t DIM>
class Size
    : public SizeBase<T, DIM, Size<T, DIM> >
{
private:
  typedef SizeBase<T, DIM, Size<T, DIM> > Base;

public:
  using Base::SizeBase;
  virtual ~Size() = default;

};

//------------------------------------------------------------------------------
/**
 * @brief The Size<T, 2> is a special size for a 2D shape defining its width and height
 */
template<typename T>
class Size<T, 2>
    : public SizeBase<T, 2, Size<T, 2> >
{
private:
  typedef SizeBase<T, 2, Size<T, 2> > Base;

public:
  using Base::SizeBase;
  virtual ~Size() = default;

  Size(const T& width, const T& height)
    : Base({width, height})
  {
  }

  /**
   * @brief width returns the width of the 2d size
   */
  T width() {return this->sz_[0];}

  /**
   * @brief height returns the length of the second dimension of the 2d size
   * @return
   */
  T height() {return this->sz_[1];}
};

//------------------------------------------------------------------------------
/**
 * @brief The Size<T, 3> is a special size for a 3D shape defining its width, height and depth
 */
template<typename T>
class Size<T, 3>
    : public SizeBase<T, 3, Size<T, 3> >
{
private:
  typedef SizeBase<T, 3, Size<T, 3> > Base;

public:
  using Base::SizeBase;
  virtual ~Size() = default;

  Size(const T& width, const T& height, const T& depth)
    : Base({width, height, depth})
  {
  }

  /**
   * @brief width returns the width of the 3d size
   */
  T width() {return this->sz_[0];}

  /**
   * @brief height returns the length of the second dimension of the 3d size
   */
  T height() {return this->sz_[1];}

  /**
   * @brief depth returns the length of the third dimension of the 3d size
   */
  T depth() {return this->sz_[2];}
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
