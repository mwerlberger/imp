#include <gtest/gtest.h>

// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <functional>
#include <limits>
#include <type_traits>

#include <imp/cu_core/cu_linearmemory.cuh>


template<class T>
typename std::enable_if<std::is_integral<T>::value, std::function<T()> >::type
getRandomGenerator()
{
  std::default_random_engine generator;
  std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::lowest(),
                                                std::numeric_limits<T>::max());
  auto random_val = std::bind(distribution, generator);
  return random_val;
}

template<class T>
typename std::enable_if<!std::is_integral<T>::value, std::function<T()> >::type
getRandomGenerator()
{
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(std::numeric_limits<T>::lowest(),
                                                 std::numeric_limits<T>::max());
  auto random_val = std::bind(distribution, generator);
  return random_val;
}


template <typename Pixel>
class CuLinearMemoryTest : public ::testing::Test
{
 protected:
  CuLinearMemoryTest()
    : linmem_(numel_)
    , linmem_copy_(numel_)
    , cu_linmem_(numel_)
  {
    using T = typename Pixel::T;
    auto random_val = getRandomGenerator<T>();

    for (size_t i=0; i<this->numel_; ++i)
    {
      linmem_[i] = random_val();
    }

    cu_linmem_.copyFrom(linmem_);
    cu_linmem_.copyTo(linmem_copy_);
  }

  size_t pixel_size_ = sizeof(Pixel);
  size_t pixel_bit_depth_ = 8*sizeof(Pixel);

  size_t numel_ = 123;
  imp::LinearMemory<Pixel> linmem_;
  imp::LinearMemory<Pixel> linmem_copy_;
  imp::cu::LinearMemory<Pixel> cu_linmem_;

};

// The list of types we want to test.
typedef testing::Types<
imp::Pixel8uC1, imp::Pixel8uC2, imp::Pixel8uC3, imp::Pixel8uC4,
imp::Pixel16uC1, imp::Pixel16uC2, imp::Pixel16uC3, imp::Pixel16uC4,
imp::Pixel32sC1, imp::Pixel32sC2, imp::Pixel32sC3, imp::Pixel32sC4,
imp::Pixel32fC1, imp::Pixel32fC2, imp::Pixel32fC3, imp::Pixel32fC4
> PixelTypes;

TYPED_TEST_CASE(CuLinearMemoryTest, PixelTypes);

TYPED_TEST(CuLinearMemoryTest, CheckMemoryAlignment)
{
  ASSERT_EQ(0, (std::uintptr_t)reinterpret_cast<void*>(this->cu_linmem_.data()) % 32);
}

TYPED_TEST(CuLinearMemoryTest, CheckLength)
{
  ASSERT_EQ(this->numel_, this->cu_linmem_.length());
}

TYPED_TEST(CuLinearMemoryTest, CheckNumBytes)
{
  ASSERT_EQ(this->numel_*this->pixel_size_, this->cu_linmem_.bytes());
}

TYPED_TEST(CuLinearMemoryTest, CheckPixelBitDepth)
{
  ASSERT_EQ(this->pixel_bit_depth_, this->cu_linmem_.bitDepth());
}

TYPED_TEST(CuLinearMemoryTest, ReturnsTrueForNonGpuMemory)
{
  ASSERT_TRUE(this->cu_linmem_.isGpuMemory());
}

TYPED_TEST(CuLinearMemoryTest, CheckMemoryWithCopy)
{
  // assumed that the copy process has already been done!
  for (size_t i=0; i<this->numel_; ++i)
    ASSERT_EQ(this->linmem_[i], this->linmem_copy_[i]);
}
