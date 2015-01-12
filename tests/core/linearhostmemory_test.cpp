// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <imp/core/linearmemory.hpp>

#include "default_msg.h"

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    size_t length = 1e4;

    // create linar hostbuffer
    imp::LinearMemory_8u_C1* h_8u_C1 = new imp::LinearMemory_8u_C1(length);
    imp::LinearMemory_32f_C1* h_32f_C1 = new imp::LinearMemory_32f_C1(length);

    // values to be set
    std::uint8_t val_8u = 1;
    float val_32f = 1.0f;
    // set host values
    *h_8u_C1 = val_8u;
    *h_32f_C1 = val_32f;

    //--------------------------------------------------------------------------
    // COPY CHECK
    {
      imp::LinearMemory_8u_C1 check_8u_C1(h_8u_C1->length());
      imp::LinearMemory_32f_C1 check_32f_C1(h_32f_C1->length());

      h_8u_C1->copyTo(check_8u_C1);
      h_32f_C1->copyTo(check_32f_C1);

      for (size_t i = 0; i<length; ++i)
      {
        assert(*h_8u_C1->data(i) == val_8u);
        assert(*h_32f_C1->data(i) == val_32f);
        assert(*check_8u_C1.data(i) == val_8u);
        assert(*check_32f_C1.data(i) == val_32f);
      }
    }

    //--------------------------------------------------------------------------
    // COPY CONSTRUCTOR CHECK
    {
      imp::LinearMemory_8u_C1 check_8u_C1(*h_8u_C1);
      imp::LinearMemory_32f_C1 check_32f_C1(*h_32f_C1);

      for (size_t i = 0; i<length; ++i)
      {
        assert(*h_8u_C1->data(i) == val_8u);
        assert(*h_32f_C1->data(i) == val_32f);
        assert(*check_8u_C1.data(i) == val_8u);
        assert(*check_32f_C1.data(i) == val_32f);
      }
    }

    //--------------------------------------------------------------------------
    // ext data pointer test
    {
      imp::LinearMemory_8u_C1 check_8u_C1(h_8u_C1->data(), h_8u_C1->length(), true);
      imp::LinearMemory_32f_C1 check_32f_C1(h_32f_C1->data(), h_32f_C1->length(), true);

      for (size_t i = 0; i<length; ++i)
      {
        assert(*h_8u_C1->data(i) == val_8u);
        assert(*h_32f_C1->data(i) == val_32f);
        assert(*check_8u_C1.data(i) == val_8u);
        assert(*check_32f_C1.data(i) == val_32f);
      }
    }

    //  cleanup
    delete(h_8u_C1);
    delete(h_32f_C1);

    std::cout << imp::ok_msg << std::endl;

  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }
  return EXIT_SUCCESS;
}
