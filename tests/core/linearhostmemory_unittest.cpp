// system includes
#include <cstdint>
#include <iostream>
#include <imp/core/linearhostmemory.h>


int main(int argc, char** argv)
{
  try
  {
    size_t length = 1e7;

    // create linar hostbuffer
    imp::LinearHostMemory_8u_C1* h_8u_C1 = new imp::LinearHostMemory_8u_C1(length);
    imp::LinearHostMemory_32f_C1* h_32f_C1 = new imp::LinearHostMemory_32f_C1(length);

    // values to be set
    std::uint8_t val_8u = 1;
    float val_32f = 1.0f;
    // set host values
    *h_8u_C1 = val_8u;
    *h_32f_C1 = val_32f;

    //--------------------------------------------------------------------------
    // COPY CHECK
    {
      imp::LinearHostMemory_8u_C1 check_8u_C1(h_8u_C1->length());
      imp::LinearHostMemory_32f_C1 check_32f_C1(h_32f_C1->length());

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
      imp::LinearHostMemory_8u_C1 check_8u_C1(*h_8u_C1);
      imp::LinearHostMemory_32f_C1 check_32f_C1(*h_32f_C1);

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
      imp::LinearHostMemory_8u_C1 check_8u_C1(h_8u_C1->data(), h_8u_C1->length(), true);
      imp::LinearHostMemory_32f_C1 check_32f_C1(h_32f_C1->data(), h_32f_C1->length(), false);

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

    std::cout << std::endl;
    std::cout << "**************************************************************************" << std::endl;
    std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
    std::cout << "**************************************************************************" << std::endl;
    std::cout << std::endl;
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    assert(false);
  }
  return EXIT_SUCCESS;
}
