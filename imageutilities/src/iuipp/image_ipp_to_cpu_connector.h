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
 * Module      : IPP-to-Ipp Connector
 * Class       : none
 * Language    : C
 * Description : Definition of some memory conversions so that an ImageIPP can be used directly instead of an ImageIpp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_IMAGE_IPP_TO_IPP_CONNECTOR_H
#define IUPRIVATE_IMAGE_IPP_TO_IPP_CONNECTOR_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iudefs.h>
#include "memorydefs_ipp.h"

namespace iuprivate {

/** Converts the ImageIpp structure to an ImageCpu type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageIpp type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
iu::ImageCpu_8u_C1* convertToCpu_8u_C1(iu::ImageIpp_8u_C1* src);
iu::ImageCpu_8u_C3* convertToCpu_8u_C3(iu::ImageIpp_8u_C3* src);
iu::ImageCpu_8u_C4* convertToCpu_8u_C4(iu::ImageIpp_8u_C4* src);
iu::ImageCpu_32f_C1* convertToCpu_32f_C1(iu::ImageIpp_32f_C1* src);
iu::ImageCpu_32f_C3* convertToCpu_32f_C3(iu::ImageIpp_32f_C3* src);
iu::ImageCpu_32f_C4* convertToCpu_32f_C4(iu::ImageIpp_32f_C4* src);

} // namespace iuprivate

#endif // IUPRIVATE_IMAGE_IPP_TO_IPP_CONNECTOR_H
