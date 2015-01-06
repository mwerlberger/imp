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
 * Module      : Core
 * Class       : none
 * Language    : C
 * Description : Definition of set value functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IMP_SETVALUE_H
#define IMP_SETVALUE_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iostream>
#include "coredefs.h"
#include "memorydefs.h"

namespace imp_detail {

///* ***************************************************************************
// *  Declaration of CUDA WRAPPERS
// * ***************************************************************************/
//extern void cuSetValue(const unsigned char& value, imp::LinearDeviceMemory_8u_C1* dst);
//extern void cuSetValue(const int& value, imp::LinearDeviceMemory_32s_C1* dst);
//extern void cuSetValue(const float& value, imp::LinearDeviceMemory_32f_C1* dst);
//extern void cuSetValue(const unsigned char& value, imp::ImageGpu_8u_C1 *dst, const IuRect &roi);
//extern void cuSetValue(const uchar2& value, imp::ImageGpu_8u_C2 *dst, const IuRect &roi);
//extern void cuSetValue(const uchar3& value, imp::ImageGpu_8u_C3 *dst, const IuRect &roi);
//extern void cuSetValue(const uchar4& value, imp::ImageGpu_8u_C4 *dst, const IuRect &roi);
//extern void cuSetValue(const int& value, imp::ImageGpu_32s_C1 *dst, const IuRect &roi);
//extern void cuSetValue(const float& value, imp::ImageGpu_32f_C1 *dst, const IuRect &roi);
//extern void cuSetValue(const float2& value, imp::ImageGpu_32f_C2 *dst, const IuRect &roi);
//extern void cuSetValue(const float3& value, imp::ImageGpu_32f_C3 *dst, const IuRect &roi);
//extern void cuSetValue(const float4& value, imp::ImageGpu_32f_C4 *dst, const IuRect &roi);
//extern void cuSetValue(const unsigned char& value, imp::VolumeGpu_8u_C1 *dst, const IuCube &roi);
//extern void cuSetValue(const uchar2& value, imp::VolumeGpu_8u_C2 *dst, const IuCube &roi);
//extern void cuSetValue(const uchar4& value, imp::VolumeGpu_8u_C4 *dst, const IuCube &roi);
//extern void cuSetValue(const unsigned short& value, imp::VolumeGpu_16u_C1 *dst, const IuCube &roi);
//extern void cuSetValue(const float& value, imp::VolumeGpu_32f_C1 *dst, const IuCube &roi);
//extern void cuSetValue(const float2& value, imp::VolumeGpu_32f_C2 *dst, const IuCube &roi);
//extern void cuSetValue(const float4& value, imp::VolumeGpu_32f_C4 *dst, const IuCube &roi);
//extern void cuSetValue(const unsigned int& value, imp::VolumeGpu_32u_C1 *dst, const IuCube &roi);
//extern void cuSetValue(const uint2& value, imp::VolumeGpu_32u_C2 *dst, const IuCube &roi);
//extern void cuSetValue(const uint4& value, imp::VolumeGpu_32u_C4 *dst, const IuCube &roi);
//extern void cuSetValue(const int& value, imp::VolumeGpu_32s_C1 *dst, const IuCube &roi);
//extern void cuSetValue(const int2& value, imp::VolumeGpu_32s_C2 *dst, const IuCube &roi);
//extern void cuSetValue(const int4& value, imp::VolumeGpu_32s_C4 *dst, const IuCube &roi);

/* ***************************************************************************/


// 1D set value; host; 8-bit
void setValue(const unsigned char& value, imp::LinearHostMemory_8u_C1* srcdst);
//void setValue(const uchar2& value, imp::LinearHostMemory_8u_C2* srcdst);
//void setValue(const uchar3& value, imp::LinearHostMemory_8u_C3* srcdst);
//void setValue(const uchar4& value, imp::LinearHostMemory_8u_C4* srcdst);

// 1D set value; host; 32-bit int
void setValue(const int& value, imp::LinearHostMemory_32s_C1* srcdst);
//void setValue(const int2& value, imp::LinearHostMemory_32s_C2* srcdst);
//void setValue(const int3& value, imp::LinearHostMemory_32s_C3* srcdst);
//void setValue(const int4& value, imp::LinearHostMemory_32s_C4* srcdst);

// 1D set value; host; 32-bit float
void setValue(const float& value, imp::LinearHostMemory_32f_C1* srcdst);
//void setValue(const float2& value, imp::LinearHostMemory_32f_C2* srcdst);
//void setValue(const float3& value, imp::LinearHostMemory_32f_C3* srcdst);
//void setValue(const float4& value, imp::LinearHostMemory_32f_C4* srcdst);

// 1D set value; device; 8-bit
//void setValue(const unsigned char& value, imp::LinearDeviceMemory_8u_C1* srcdst);
//void setValue(const uchar2& value, imp::LinearDeviceMemory_8u_C2* srcdst);
//void setValue(const uchar3& value, imp::LinearDeviceMemory_8u_C3* srcdst);
//void setValue(const uchar4& value, imp::LinearDeviceMemory_8u_C4* srcdst);

// 1D set value; device; 32-bit int
//void setValue(const int& value, imp::LinearDeviceMemory_32s_C1* srcdst);
//void setValue(const int2& value, imp::LinearDeviceMemory_32s_C2* srcdst);
//void setValue(const int3& value, imp::LinearDeviceMemory_32s_C3* srcdst);
//void setValue(const int4& value, imp::LinearDeviceMemory_32s_C4* srcdst);

// 1D set value; device; 32-bit float
//void setValue(const float& value, imp::LinearDeviceMemory_32f_C1* srcdst);
//void setValue(const float2& value, imp::LinearDeviceMemory_32f_C2* srcdst);
//void setValue(const float3& value, imp::LinearDeviceMemory_32f_C3* srcdst);
//void setValue(const float4& value, imp::LinearDeviceMemory_32f_C4* srcdst);

//// 2D set pixel value; host;
//template<typename PixelType, class Allocator, IuPixelType _pixel_type>
//inline void setValue(const PixelType &value,
//                     imp::ImageCpu<PixelType, Allocator, _pixel_type> *srcdst,
//                     const IuRect& roi)
//{
//  for(unsigned int y=roi.y; y<roi.height; ++y)
//  {
//    for(unsigned int x=roi.x; x<roi.width; ++x)
//    {
//      *srcdst->data(x,y) = value;
//    }
//  }
//}

//// 3D set pixel value; host;
//template<typename PixelType, class Allocator, IuPixelType _pixel_type>
//inline void setValue(const PixelType &value,
//                     imp::VolumeCpu<PixelType, Allocator, _pixel_type> *srcdst,
//                     const IuCube& roi)
//{
//  for(unsigned int z=roi.z; z<roi.depth; ++z)
//  {
//    for(unsigned int y=roi.y; y<roi.height; ++y)
//    {
//      for(unsigned int x=roi.x; x<roi.width; ++x)
//      {
//        *srcdst->data(x,y,z) = value;
//      }
//    }
//  }
//}

//// 2D set pixel value; device;
//template<typename PixelType, class Allocator, IuPixelType _pixel_type>
//void setValue(const PixelType &value, imp::ImageGpu<PixelType, Allocator, _pixel_type> *srcdst, const IuRect& roi)
//{
//  cuSetValue(value, srcdst, roi);
//}

//// 3D set pixel value; device;
//template<typename PixelType, class Allocator, IuPixelType _pixel_type>
//void setValue(const PixelType &value,
//              imp::VolumeGpu<PixelType, Allocator, _pixel_type> *srcdst,
//              const IuCube& roi)
//{
//  cuSetValue(value, srcdst, roi);
//}


} // namespace iuprivate

#endif // IMP_SETVALUE_H
