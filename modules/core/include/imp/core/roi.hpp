#ifndef IMP_ROI_HPP
#define IMP_ROI_HPP

#include <imp/core/size.hpp>

namespace imp {

class Roi
{
public:
  Roi() :
      x(0), y(0), z(0), width(0), height(0), depth(0)
  {
  }

  Roi(int _x, int _y, int _z, unsigned int _width, unsigned int _height, unsigned int _depth) :
      x(_x), y(_y), z(_z), width(_width), height(_height), depth(_depth)
  {
  }

  Roi(const Roi& from) :
      x(from.x), y(from.y), z(from.z), width(from.width), height(from.height), depth(from.depth)
  {
  }

  Roi& operator= (const Roi& from)
  {
//    if (from == *this)
//      return *this;

    this->x = from.x;
    this->y = from.y;
    this->z = from.z;
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;

    return *this;
  }

  Roi(const IuSize& from) :
      x(0), y(0), z(0), width(from.width), height(from.height), depth(from.depth)
  {
  }

  Roi& operator= (const IuSize& from)
  {
    this->x = 0;
    this->y = 0;
    this->z = 0;
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;

    return *this;
  }

public:
  int x;       //!< x-coord of the upper left corner
  int y;       //!< y-coord of the upper left corner
  int z;       //!< z-coord of the upper left corner
  unsigned int width;   //!< width of the rectangle
  unsigned int height;  //!< height of the rectangle
  unsigned int depth;  //!< depth of the rectangle

};

}

#endif // imp_ROI_HPP

