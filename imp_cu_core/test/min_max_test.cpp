#include <gtest/gtest.h>

// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <functional>

#include <imp/core/image_raw.hpp>
#include <imp/cu_core/cu_math.cuh>
#include <imp/cu_core/cu_utils.hpp>


TEST(IMPCuCoreTestSuite,minMaxTest)
{
  // setup random number generator
  std::default_random_engine generator;
  std::uniform_int_distribution<std::uint8_t> distribution(0, UINT8_MAX);
  auto random_uint8 = std::bind(distribution, generator);

  imp::size_type width = 123;
  imp::size_type height = 324;
  imp::ImageRaw8uC1 im(width,height);
  std::uint8_t min_val = UINT8_MAX, max_val = 0;
  for (imp::size_type y=0; y<height; ++y)
  {
    for (imp::size_type x=0; x<width; ++x)
    {
      std::uint8_t random_value = random_uint8();
      im[y][x] = random_value;
      min_val = std::min(min_val, random_value);
      max_val = std::max(max_val, random_value);
    }
  }

  IMP_CUDA_CHECK();
  imp::cu::ImageGpu8uC1 cu_im(im);
  IMP_CUDA_CHECK();
  imp::Pixel8uC1 min_pixel, max_pixel;
  IMP_CUDA_CHECK();
  imp::cu::minMax(cu_im, min_pixel, max_pixel);
  IMP_CUDA_CHECK();

  ASSERT_EQ(min_val, min_pixel);
  ASSERT_EQ(max_val, max_pixel);
}
