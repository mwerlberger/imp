#include <gtest/gtest.h>

// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>

#include <imp/core/roi.hpp>


TEST(IMPCoreTestSuite,roiTest)
{
  //
  // 2D case
  //
  std::int32_t x=1, y=2, w=10, h=13;
  imp::Roi2i roi(x,y,w,h);
  ASSERT_TRUE(x == roi.x());
  ASSERT_TRUE(y == roi.y());
  ASSERT_TRUE(w == roi.width());
  ASSERT_TRUE(h == roi.height());

  ASSERT_TRUE(x == roi.lu()[0]);
  ASSERT_TRUE(y == roi.lu()[1]);
  ASSERT_TRUE(w == roi.size()[0]);
  ASSERT_TRUE(h == roi.size()[1]);
}
