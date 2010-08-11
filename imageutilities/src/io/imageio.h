/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : IO
 * Class       : none
 * Language    : C++
 * Description : Definition of image I/O functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUIO_IMAGEIO_H
#define IUIO_IMAGEIO_H

#include <core/coredefs.h>
#include <core/memorydefs.h>
#include <string>

namespace iuprivate {

iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename);
iu::ImageNpp_32f_C1* imread_cu32f_C1(const std::string& filename);

bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename);
bool imsave(iu::ImageNpp_32f_C1* image, const std::string& filename);

} // namespace iuprivate


#endif // IUIO_IMAGEIO_H
