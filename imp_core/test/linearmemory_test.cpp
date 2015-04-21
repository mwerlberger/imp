#include <gtest/gtest.h>

// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>

#include <imp/core/linearmemory.hpp>


template <typename Pixel>
class LinearMemoryTest : public ::testing::Test
{
 protected:
  LinearMemoryTest() :
    linmem_(numel_)
  {

  }

  size_t pixel_size_ = sizeof(Pixel);
  size_t pixel_bit_depth_ = 8*sizeof(Pixel);

  size_t numel_ = 123;
  imp::LinearMemory<Pixel> linmem_;
};

// The list of types we want to test.
typedef testing::Types<
imp::Pixel8uC1, imp::Pixel8uC2, imp::Pixel8uC3, imp::Pixel8uC4,
imp::Pixel16uC1, imp::Pixel16uC2, imp::Pixel16uC3, imp::Pixel16uC4,
imp::Pixel32sC1, imp::Pixel32sC2, imp::Pixel32sC3, imp::Pixel32sC4,
imp::Pixel32fC1, imp::Pixel32fC2, imp::Pixel32fC3, imp::Pixel32fC4> PixelTypes;

TYPED_TEST_CASE(LinearMemoryTest, PixelTypes);

TYPED_TEST(LinearMemoryTest, CheckMemoryAlignment)
{
  ASSERT_EQ(0, (std::uintptr_t)reinterpret_cast<void*>(this->linmem_.data()) % 32);
}

TYPED_TEST(LinearMemoryTest, CheckLength)
{
  ASSERT_EQ(this->numel_, this->linmem_.length());
}

TYPED_TEST(LinearMemoryTest, CheckNumBytes)
{
  ASSERT_EQ(this->numel_*this->pixel_size_, this->linmem_.bytes());
}

TYPED_TEST(LinearMemoryTest, CheckPixelBitDepth)
{
  ASSERT_EQ(this->pixel_bit_depth_, this->linmem_.bitDepth());
}

TYPED_TEST(LinearMemoryTest, ReturnsFalseForNonGpuMemory)
{
  ASSERT_FALSE(this->linmem_.isGpuMemory());
}
