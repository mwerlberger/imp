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
 * Description : Implementation of set value functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <cstring>
#include <imp/core/setvalue.h>

namespace imp_detail {

//-----------------------------------------------------------------------------
// [1D; host] set values; 8-bit
void setValue(const unsigned char& value, imp::LinearHostMemory_8u_C1* srcdst)
{
  memset((void*)srcdst->data(), value, srcdst->bytes());
}

//-----------------------------------------------------------------------------
// [1D; host] set values; 32-bit
void setValue(const int& value, imp::LinearHostMemory_32s_C1* srcdst)
{
  // we are using for loops because memset is only safe on integer type arrays

  int* buffer = srcdst->data();
  for(unsigned int i=0; i<srcdst->length(); ++i)
  {
    buffer[i] = value;
  }
}


//-----------------------------------------------------------------------------
// [1D; host] set values; 32-bit
void setValue(const float& value, imp::LinearHostMemory_32f_C1* srcdst)
{
  // we are using for loops because memset is only safe on integer type arrays

  float* buffer = srcdst->data();
  for(unsigned int i=0; i<srcdst->length(); ++i)
  {
    buffer[i] = value;
  }
}

////-----------------------------------------------------------------------------
//// [1D; device] set values; 8-bit
//void setValue(const unsigned char& value, imp::LinearDeviceMemory_8u_C1* srcdst)
//{
//  // cudaMemset is slow so we are firing up a kernel
//  cuSetValue(value, srcdst);
//}

//// [1D; device] set values; 32-bit
//void setValue(const int& value, imp::LinearDeviceMemory_32s_C1* srcdst)
//{
//  // cudaMemset is slow so we are firing up a kernel
//  cuSetValue(value, srcdst);
//}

//// [1D; device] set values; 32-bit
//void setValue(const float& value, imp::LinearDeviceMemory_32f_C1* srcdst)
//{
//  // cudaMemset is slow so we are firing up a kernel
//  cuSetValue(value, srcdst);
//}

} // namespace imp_detail
