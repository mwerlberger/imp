#include <stdlib.h>
#include <iostream>
#include <cstdint>
#include <glog/logging.h>
#include <imp/core/timer.hpp>
#include <imp/core/image_raw.hpp>

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  VLOG(2) << "Starting aligned allocator benchmarking";


  const size_t memory_size = 1e6;
  int memaddr_align = 32;

  imp::SingleShotTimer timer_posix_memalign("posix_memalign");

  for (std::uint64_t i=0; i<1e6; ++i)
  {
    std::uint8_t* p_data_aligned;
    posix_memalign((void**)&p_data_aligned, memaddr_align, memory_size);

  }

}
