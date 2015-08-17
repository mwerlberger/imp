// system includes
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <imp/core/linearmemory.hpp>
#include <imp/cu_core/cu_linearmemory.cuh>

#include <cuda_runtime.h>

#include "default_msg.h"

int main(int /*argc*/, char** /*argv*/)
{
  try
  {
    size_t length = 1e4;

    // create linar buffers.... (host and device)
    std::cout << "alloc 8uC1 device memory" << std::endl;
    imp::cu::LinearMemory8uC1* d_8u_C1 = new imp::cu::LinearMemory8uC1(length);
    std::cout << "alloc 32fC1 device memory" << std::endl;
    imp::cu::LinearMemory32fC1* d_32f_C1 = new imp::cu::LinearMemory32fC1(length);

    imp::LinearMemory8uC1* h_8u_C1 = new imp::LinearMemory8uC1(length);
    imp::LinearMemory32fC1* h_32f_C1 = new imp::LinearMemory32fC1(length);

    // values to be set (on host)
    imp::Pixel8uC1 val_8u = 1;
    imp::Pixel32fC1 val_32f = 1.0f;
    // set host values
    *h_8u_C1 = val_8u;
    *h_32f_C1 = val_32f;

    // copy host->device
    d_8u_C1->copyFrom(*h_8u_C1);
    d_32f_C1->copyFrom(*h_32f_C1);


    //--------------------------------------------------------------------------
    // COPY CHECK device->device->host
    {
      imp::cu::LinearMemory8uC1 d_copy_8u_C1(d_8u_C1->length());
      imp::cu::LinearMemory32fC1 d_copy_32f_C1(d_32f_C1->length());

      imp::LinearMemory8uC1 check_8u_C1(d_copy_8u_C1.length());
      imp::LinearMemory32fC1 check_32f_C1(d_copy_32f_C1.length());

      d_8u_C1->copyTo(d_copy_8u_C1);
      d_32f_C1->copyTo(d_copy_32f_C1);

      d_copy_8u_C1.copyTo(check_8u_C1);
      d_copy_32f_C1.copyTo(check_32f_C1);

      for (size_t i = 0; i<length; ++i)
      {
        assert(*h_8u_C1->data(i) == val_8u);
        assert(*h_32f_C1->data(i) == val_32f);
        assert(*check_8u_C1.data(i) == val_8u);
        assert(*check_32f_C1.data(i) == val_32f);
      }
    }

    //--------------------------------------------------------------------------
    // COPY CONSTRUCTOR CHECK device->device
    {
      imp::cu::LinearMemory8uC1 d_copy_8u_C1(*d_8u_C1);
      imp::cu::LinearMemory32fC1 d_copy_32f_C1(*d_32f_C1);

      imp::LinearMemory8uC1 check_8u_C1(d_copy_8u_C1.length());
      imp::LinearMemory32fC1 check_32f_C1(d_copy_32f_C1.length());

      d_copy_8u_C1.copyTo(check_8u_C1);
      d_copy_32f_C1.copyTo(check_32f_C1);

      for (size_t i = 0; i<length; ++i)
      {
        assert(*h_8u_C1->data(i) == val_8u);
        assert(*h_32f_C1->data(i) == val_32f);
        assert(*check_8u_C1.data(i) == val_8u);
        assert(*check_32f_C1.data(i) == val_32f);
      }
    }

    //--------------------------------------------------------------------------
    // COPY CONSTRUCTOR CHECK host->device
    {
      imp::cu::LinearMemory8uC1 d_copy_8u_C1(*h_8u_C1);
      imp::cu::LinearMemory32fC1 d_copy_32f_C1(*h_32f_C1);

      imp::LinearMemory8uC1 check_8u_C1(d_copy_8u_C1.length());
      imp::LinearMemory32fC1 check_32f_C1(d_copy_32f_C1.length());

      d_copy_8u_C1.copyTo(check_8u_C1);
      d_copy_32f_C1.copyTo(check_32f_C1);

      for (size_t i = 0; i<length; ++i)
      {
        assert(*h_8u_C1->data(i) == val_8u);
        assert(*h_32f_C1->data(i) == val_32f);
        assert(*check_8u_C1.data(i) == val_8u);
        assert(*check_32f_C1.data(i) == val_32f);
      }
    }

    //    //--------------------------------------------------------------------------
    //    // ext data pointer test
    //    {
    //      imp::LinearMemory8uC1 check_8u_C1(h_8u_C1->data(), h_8u_C1->length(), true);
    //      imp::LinearMemory32fC1 check_32f_C1(h_32f_C1->data(), h_32f_C1->length(), true);

    //      for (size_t i = 0; i<length; ++i)
    //      {
    //        assert(*h_8u_C1->data(i) == val_8u);
    //        assert(*h_32f_C1->data(i) == val_32f);
    //        assert(*check_8u_C1.data(i) == val_8u);
    //        assert(*check_32f_C1.data(i) == val_32f);
    //      }
    //    }

    //--------------------------------------------------------------------------
    //  cleanup
    std::cout << "deleting 8uC1 device memory" << std::endl;
    delete(d_8u_C1);
    std::cout << "deleting 32fC1 device memory" << std::endl;
    delete(d_32f_C1);

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
