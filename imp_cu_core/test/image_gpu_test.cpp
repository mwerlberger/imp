#include <gtest/gtest.h>

// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>

#include <imp/cu_core/cu_image_gpu.cuh>


template <typename Pixel, imp::PixelType pixel_type>
class ImageGpuTest : public ::testing::Test
{
 protected:
  ImageGpuTest() :
    im_(numel_)
  {
  }

  size_t pixel_size_ = sizeof(Pixel);
  size_t pixel_bit_depth_ = 8*sizeof(Pixel);

  size_t width = 256;
  size_t height = 256;
  imp::ImageGpu<Pixel> im_;
};

// The list of types we want to test.
typedef testing::Types<
imp::Pixel8uC1, imp::Pixel8uC2, imp::Pixel8uC3, imp::Pixel8uC4,
imp::Pixel16uC1, imp::Pixel16uC2, imp::Pixel16uC3, imp::Pixel16uC4,
imp::Pixel32sC1, imp::Pixel32sC2, imp::Pixel32sC3, imp::Pixel32sC4,
imp::Pixel32fC1, imp::Pixel32fC2, imp::Pixel32fC3, imp::Pixel32fC4> PixelTypes;

TYPED_TEST_CASE(ImageGpuTest, PixelTypes);

TYPED_TEST(ImageGpuTest, CheckMemoryAlignment)
{
  ASSERT_EQ(0, (std::uintptr_t)reinterpret_cast<void*>(this->im_.data()) % 32);
}

TYPED_TEST(ImageGpuTest, CheckLength)
{
  ASSERT_EQ(this->numel_, this->im_.length());
}

TYPED_TEST(ImageGpuTest, CheckNumBytes)
{
  ASSERT_EQ(this->numel_*this->pixel_size_, this->im_.bytes());
}

TYPED_TEST(ImageGpuTest, CheckPixelBitDepth)
{
  ASSERT_EQ(this->pixel_bit_depth_, this->im_.bitDepth());
}

TYPED_TEST(ImageGpuTest, ReturnsFalseForNonGpuMemory)
{
  ASSERT_FALSE(this->im_.isGpuMemory());
}
