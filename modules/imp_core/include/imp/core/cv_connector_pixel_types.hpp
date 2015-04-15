#ifndef IMP_CV_CONNECTOR_PIXEL_TYPES
#define IMP_CV_CONNECTOR_PIXEL_TYPES

#include <imp/core/pixel_enums.hpp>

namespace imp {

imp::PixelType pixelTypeFromCv(int type);
int pixelTypeToCv(imp::PixelType type);


} // namespace imp

#endif // IMP_CV_CONNECTOR_PIXEL_TYPES

