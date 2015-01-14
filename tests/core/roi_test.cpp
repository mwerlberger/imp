// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <imp/core/roi.hpp>

#include "default_msg.h"


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    //
    // 2D case
    //
    std::int32_t x=1, y=2, w=10, h=13;
    imp::Roi2i roi(x,y,w,h);
    assert(x == roi.x());
    assert(y == roi.y());
    assert(w == roi.width());
    assert(h == roi.height());

    assert(x == roi.lu()[0]);
    assert(y == roi.lu()[1]);
    assert(w == roi.size()[0]);
    assert(h == roi.size()[1]);

    std::cout << "Roi (dim=" << static_cast<int>(roi.dim()) << "): ("
              << roi.x() << "," << roi.y() << ") "
              << roi.width() << "x" << roi.height() << std::endl;

    // comparison operator tests

//    imp::Roi2i a(123,456);
//    imp::Roi2i b(123,456);
//    imp::Roi2i c(124,456);
//    imp::Roi2i d(124,457);

//    assert(true  == (a == b));
//    assert(false == (a != b));
//    assert(false == (a == c));
//    assert(true  == (a != c));

  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }

  std::cout << imp::ok_msg << std::endl;

  return EXIT_SUCCESS;

}
