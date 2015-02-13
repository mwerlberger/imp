#ifndef IMP_TYPES_HPP
#define IMP_TYPES_HPP

#include <cstdint>

namespace imp {

// typedefs
typedef std::uint32_t size_type;

enum class InterpolationMode
{
  point,
  linear,
  cubic,
  cubicSpline
};


}

#endif // IMP_TYPES_HPP

