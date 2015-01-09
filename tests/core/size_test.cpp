// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <imp/core/size.hpp>


int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    std::int32_t w=10, h=13;
    std::array<decltype(w), 2> bla = {w, h};
    imp::Size2i sz(bla);
//    assert(w == sz.width());
//    assert(h == sz.width());

    assert(w == sz.data()[0]);
    assert(h == sz.data()[1]);

//    std::cout << "size of dimention " << sz.dim() << " with values: "
//              << sz.width() << ", " << sz.height() << std::endl;
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }
  return EXIT_SUCCESS;

}
