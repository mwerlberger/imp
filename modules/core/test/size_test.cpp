#include <gtest/gtest.h>

// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>

#include <imp/core/size.hpp>

TEST(IMPCoreTestSuite,testSize)
{
  imp::Size2u sz2u;
  EXPECT_EQ(0, sz2u.width());
  EXPECT_EQ(0, sz2u.height());

  // 2D sizes
  std::int32_t w=10, h=13;
  imp::Size2i sz(w,h);
  EXPECT_EQ(w, sz.width());
  EXPECT_EQ(h, sz.height());

  EXPECT_EQ(w, sz.data()[0]);
  EXPECT_EQ(h, sz.data()[1]);

  // comparison operator tests
  imp::Size2i a(123,456);
  imp::Size2i b(123,456);
  imp::Size2i c(124,456);
  imp::Size2i d(124,457);

  ASSERT_TRUE((a == b));
  ASSERT_FALSE((a != b));
  ASSERT_FALSE((a == c));
  ASSERT_TRUE((a != c));
}
