#ifndef IMP_SIZE_HPP
#define IMP_SIZE_HPP

#include <cstdint>
#include <array>
#include <algorithm>

namespace imp {

/**
 * @brief The class Size defines the templated base class of any size
 */
template<typename T, std::uint8_t DIM>
class Size
{
public:
  Size()
  {
    std::fill(sz_.begin(), sz_.end(), 0);
  }

  Size(const std::array<T,DIM>& arr)
    : sz_ (arr)
  {
  }

  virtual ~Size()
  {
    // nothing to do here
  }

  Size(const Size& from)
    : sz_(from.sz_)
  {
  }

  Size& operator= (const Size& from)
  {
    this->sz_ = from.sz_;
  }

  /**
   * @brief dim Returns the dimension of the size object.
   * @return Dimension.
   */
  std::uint8_t dim() const {return DIM;}

  /**
   * @brief data gives access to the underlying (raw) element storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  T* data() {return sz_.data();}

  /**
   * @brief data gives access to the underlying (raw) const element storage
   * @return Pointer address to the buffer of the underlying data storage.
   */
  const T* data() const {return reinterpret_cast<const T*>(sz_.data());}

protected:
  std::array<T, DIM> sz_;
};

///**
// * template specializiation of the size of a 2D shape.
// */
//template<typename T>
//class Size<T, 2>
//{
//public:
//  Size(const T& width, const T& height)
//  {
//    this->sz_ = {width, height};
//  }

//  /**
//   * @brief width returns the width of the 2d size
//   */
//  T width() {return this->sz_[0];}
//  /**
//   * @brief height returns the length of the second dimension of the 2d size
//   * @return
//   */
//  T height() {return this->sz_[1];}

//  std::uint8_t dim() const;
//};


// some convencience typedefs

typedef Size<std::uint32_t, 2> Size2u;
typedef Size<std::int32_t, 2> Size2i;
typedef Size<float, 2> Size2f;



//namespace deprecated {
///**
// * @brief The Size class defines width, height and optinally also depth
// */
//class Size
//{
//public:
//  Size() :
//      width(0)
//    , height(0)
//    , depth(0)
//  {
//  }

//  Size(std::uint32_t _width, std::uint32_t _height, std::uint32_t _depth = 1) :
//      width(_width)
//    , height(_height)
//    , depth(_depth)
//  {
//  }

//  Size(const Size& from) :
//      width(from.width)
//    , height(from.height)
//    , depth(from.depth)
//  {
//  }

//  Size& operator= (const Size& from)
//  {
////    if(from == *this)
////      return *this;

//    this->width = from.width;
//    this->height = from.height;
//    this->depth = from.depth;
//    return *this;
//  }


//  Size operator* (const double factor) const
//  {
//    return Size(static_cast<std::uint32_t>(this->width * factor + 0.5f),
//                static_cast<std::uint32_t>(this->height * factor + 0.5f),
//                static_cast<std::uint32_t>(this->depth * factor + 0.5f));
//  }

//  Size operator/ (const double factor) const
//  {
//    IU_ASSERT(factor != 0);
//    double invFactor = 1 / factor;
//    return Size(this->width, this->height, this->depth) * invFactor;
//  }

//public:
//  std::uint32_t width;
//  std::uint32_t height;
//  std::uint32_t depth;
//};

//inline bool operator==(const Size& lhs, const Size& rhs)
//{
//  return ((lhs.width == rhs.width) && (lhs.height == rhs.height) && (lhs.depth == rhs.depth));
//}

//inline bool operator!=(const Size& lhs, const Size& rhs)
//{
//  return ((lhs.width != rhs.width) || (lhs.height != rhs.height) || (lhs.depth != rhs.depth));
//}

//} // namespace deprecated

} // namespace imp

#endif // IMP_SIZE_HPP
