// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <imp/core/size.hpp>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    //
    // 2D case
    //
    std::int32_t w=10, h=13;
    imp::Size2i sz(w,h);
    assert(w == sz.width());
    assert(h == sz.height());

    assert(w == sz.data()[0]);
    assert(h == sz.data()[1]);

//    std::cout << "size of dimension " << static_cast<int>(sz1.dim()) << " with values: "
//              << sz1.width() << ", " << sz1.height() << std::endl;

    // comparison operator tests

    imp::Size2i a(123,456);
    imp::Size2i b(123,456);
    imp::Size2i c(124,456);
    imp::Size2i d(124,457);

    assert(true  == (a == b));
    assert(false == (a != b));
    assert(false == (a == c));
    assert(true  == (a != c));

//    assert(false == (a >  c));
//    assert(false == (a >= c));
//    assert(false == (a <  c));
//    assert(true  == (a <= c));

//    assert(false == (a >  d));
//    assert(false == (a >= d));
//    assert(true  == (a <  d));
//    assert(true  == (a <= d));

  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  std::cout << imp::ok_msg << std::endl;

  return EXIT_SUCCESS;

}
