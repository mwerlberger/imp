#ifndef IMP_ROI_HPP
#define IMP_ROI_HPP

#include <cstdint>
#include <array>
#include <algorithm>

#include <imp/core/size.hpp>

namespace imp {

//------------------------------------------------------------------------------
/**
 * @brief The class RoiBase defines the templated base class of an \a DIM-dimensional
 *        region-of-interest utilizing the CRTP pattern.
 */
template<typename T, std::uint8_t DIM, typename Derived>
struct RoiBase
{
  std::array<T, DIM> pt; //!< internal data storage for all dimensions' 'left-upper' corner
  imp::Size<T, DIM> sz; //!< size of the given roi

  RoiBase()
  {
    std::fill(pt.begin(), pt.end(), 0);
  }

  /**
   * @brief RoiBase Constructor given the lef-upper corner and the ROI's size
   * @param lu array of the left-upper corner of the format {a1, a2, a3, ...., aN}
   * @param sz Size of the \a DIM-dimensional ROI
   */
  RoiBase(const std::array<T,DIM>& lu, const imp::Size<T,DIM>& sz)
    : pt(lu)
    , sz(sz)
  {
  }

  /**
   * @brief RoiBase initialized max size only. The top-left corner will be ZEROS
   * @param sz The ROI's size in all dimensions.
   */
  RoiBase(const imp::Size<T,DIM>& sz)
    : sz(sz)
  {
    std::fill(pt.begin(), pt.end(), 0);
  }

  virtual ~RoiBase() = default;

  RoiBase(const RoiBase& from)
    : pt(from.pt)
    , sz(from.sz)
  {
  }

  RoiBase& operator= (const RoiBase& from)
  {
    this->sz = from.sz;
    this->pt = from.pt;
    return *this;
  }

  /**
   * @brief dim Returns the dimension of the Roi object.
   * @return Dimension.
   */
  std::uint8_t dim() const {return DIM;}

  /**
   * @brief data gives access to the underlying (raw) data storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  T* luRaw() {return pt.data();}

  /**
   * @brief data gives access to the underlying (raw) const data storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  const T* luRaw() const {return reinterpret_cast<const T*>(pt.data());}

  /**
   * @brief array with the internal data storing the values for the left-upper corner
   */
  std::array<T, DIM>& lu() {return pt;}
  /**
   * @brief const array with the internal data storing the values for the left-upper corner
   */
  const std::array<T, DIM>& lu() const {return reinterpret_cast<const std::array<T, DIM>&>(pt);}

  /**
   * @brief size of the ROI
   */
  imp::Size<T, DIM>& size() {return sz;}
  /**
   * @brief size of the ROI (const)
   */
  const imp::Size<T, DIM>& size() const {return reinterpret_cast<const imp::Size<T, DIM>&>(sz);}

};

//------------------------------------------------------------------------------
// relational operators

template<typename T, std::uint8_t DIM, typename Derived>
inline bool operator==(const RoiBase<T, DIM, Derived>& lhs,
                       const RoiBase<T, DIM, Derived>& rhs)
{
  return ( (lhs.pt() == rhs.pt()) && (lhs.sz() == rhs.sz()) );
}

template<typename T, std::uint8_t DIM, typename Derived>
inline bool operator!=(const RoiBase<T, DIM, Derived>& lhs,
                       const RoiBase<T, DIM, Derived>& rhs)
{
  return ( (lhs.pt() != rhs.pt()) || (lhs.sz() != rhs.sz()) );
}

//------------------------------------------------------------------------------
/**
 * @brief The class Roi defines region of interest for \a DIM dimensions
 */
template<typename T, std::uint8_t DIM>
struct Roi
    : public RoiBase<T, DIM, Roi<T, DIM> >
{
  typedef RoiBase<T, DIM, Roi<T, DIM> > Base;

  using Base::RoiBase;
  virtual ~Roi() = default;
};

//------------------------------------------------------------------------------
/**
 * @brief The Roi<T, 2> is a special Roi for a 2D shape defining its width and height
 */
template<typename T>
struct Roi<T, 2>
    : public RoiBase<T, 2, Roi<T, 2> >
{
  typedef RoiBase<T, 2, Roi<T, 2> > Base;

  using Base::RoiBase;
  virtual ~Roi() = default;

  Roi(const T& x, const T& y, const T& width, const T& height)
    : Base({x,y}, {width, height})
  {
  }

  /**
   * @brief x returns the ROI's x coordinate of the left-upper corner
   */
  T x() const {return this->pt[0];}

  /**
   * @brief y returns the ROI's y coordinate of the left-upper corner
   */
  T y() const {return this->pt[1];}

  /**
   * @brief width returns the width of the 2d Roi
   */
  T width() const {return this->sz[0];}

  /**
   * @brief height returns the length of the second dimension of the 2d Roi
   * @return
   */
  T height() const {return this->sz[1];}
};


//------------------------------------------------------------------------------
// some convencience typedefs

// 2D
typedef Roi<std::uint32_t, 2> Roi2u;
typedef Roi<std::int32_t, 2> Roi2i;
typedef Roi<float, 2> Roi2f;
typedef Roi<float, 2> Roi2d;
//3D
typedef Roi<std::uint32_t, 3> Roi3u;
typedef Roi<std::int32_t, 3> Roi3i;
typedef Roi<float, 3> Roi3f;
typedef Roi<float, 3> Roi3d;


} // namespace imp

#endif // IMP_ROI_HPP

