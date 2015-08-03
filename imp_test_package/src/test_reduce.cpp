#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cuda_runtime.h>
#include <math.h>       /* sqrt */
#include <cmath> /* std::floor */
#include <ctime>
#include <imp/imp_test_package/kernel_reduce.hpp>
#include <imp/imp_test_package/helper_cuda.h>

#define cudaCheckError() {                                                                       \
  cudaError_t e=cudaGetLastError();                                                        \
  if(e!=cudaSuccess) {                                                                     \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
  exit(EXIT_FAILURE);                                                                  \
  }                                                                                        \
  }

#define CLOCK_TO_MS(C_START,C_END) 1000.0 * (C_END - C_START) / CLOCKS_PER_SEC


unsigned int nextPow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

extern "C"
bool isPow2(unsigned int x)
{
  return ((x&(x-1))==0);
}

// elements_oer_tread is 2, or std::floor(log2 (static_cast<double>(nr_elements)))
void getNumBlocksAndThreads(int nr_elements, int max_blocks, int max_threads, int elements_per_thread, int &blocks, int &threads)
{

  //get device capability, to avoid block/grid size excceed the upbound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  threads = (nr_elements < max_threads*2) ? nextPow2((nr_elements + 1)/ 2) : max_threads;
  blocks = (nr_elements + (threads * elements_per_thread - 1)) / (threads * elements_per_thread);

  if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
  {
    printf("n is too large, please choose a smaller number!\n");
  }

  if(blocks > max_blocks)
  {
    blocks = max_blocks;
  }

  if (blocks > prop.maxGridSize[0])
  {
    printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
           blocks, prop.maxGridSize[0], threads*2, threads);

    blocks /= 2;
    threads *= 2;
  }
}

void initTestData(int* data,const unsigned int nr_elements)
{
  for (unsigned int i=0; i<nr_elements; i++)
  {
    data[i] = (rand() & 0xFF);
  }
}

void initTestData(float* data,const unsigned int nr_elements)
{
  for (unsigned int i=0; i<nr_elements; i++)
  {
    data[i] = (rand() & 0xFF) / (float)RAND_MAX;
  }
}

template<typename T>
void reductionBenchmarkSum(int _nr_elements)
{
  unsigned int nr_elements = _nr_elements;
  unsigned int nr_elements_bytes = nr_elements*sizeof(T);
  unsigned int nr_kernel_calls = 100;
  int max_threads = 256;
  int max_blocks = 64;
  //int nr_elements_per_thread = 2;
  int nr_elements_per_thread = std::floor(log2 (static_cast<double>(nr_elements)));

  T *h_idata = (T *) malloc(nr_elements_bytes);
  initTestData(h_idata,nr_elements);

  int num_blocks = 0;
  int num_threads = 0;
  getNumBlocksAndThreads(nr_elements, max_blocks, max_threads, nr_elements_per_thread, num_blocks, num_threads);

  std::cout << "Nr blocks: " << num_blocks << std::endl;
  std::cout << "Nr threads: " << num_threads << std::endl;

  // allocate mem for the result on host side
  T *h_odata = (T *) malloc(num_blocks*sizeof(T));

  // allocate device memory and data
  T *d_idata = NULL;
  T *d_odata = NULL;

  checkCudaErrors(cudaMalloc((void **) &d_idata, nr_elements_bytes));
  checkCudaErrors(cudaMalloc((void **) &d_odata, num_blocks*sizeof(T)));

  // copy data directly to device memory
  checkCudaErrors(cudaMemcpy(d_idata, h_idata, nr_elements_bytes, cudaMemcpyHostToDevice));

  // warm up cuda call
  reduce<T>(nr_elements, num_threads, num_blocks, d_idata, d_odata);

  std::clock_t c_start_gpu = std::clock();
  T sum_from_gpu = 0;
  for(unsigned int ii = 0; ii < nr_kernel_calls; ++ii)
  {
    reduce<T>(nr_elements, num_threads, num_blocks, d_idata, d_odata);
    cudaDeviceSynchronize();

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, num_blocks*sizeof(T), cudaMemcpyDeviceToHost));

    //check result
    sum_from_gpu = 0;
    for(int ii = 0; ii < num_blocks;++ii)
    {
      sum_from_gpu += h_odata[ii];
    }
  }
  std::clock_t c_end_gpu = std::clock();
  double time_gpu_ms = CLOCK_TO_MS(c_start_gpu,c_end_gpu)/((double) nr_kernel_calls);

  T sum_from_cpu = 0;
  std::clock_t c_start_cpu = std::clock();
  for(unsigned int ii = 0; ii < nr_elements; ++ii)
  {
    sum_from_cpu += h_idata[ii];
  }
  std::clock_t c_end_cpu = std::clock();
  double time_cpu_ms = CLOCK_TO_MS(c_start_cpu,c_end_cpu);

  std::cout << "sum GPU: " << sum_from_gpu << " Time per kernel " << time_gpu_ms << " Troughput " << (1.0e-9 * ((double) nr_elements_bytes))/(time_gpu_ms/1000) << " GB/s " << std::endl;
  std::cout << "sum CPU: " << sum_from_cpu << " Time " << time_cpu_ms << std::endl;

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
}

void reductionBenchmarkHessian(int _nr_elements)
{

}

int main(int argc, const char* argv[]) {

  //------------ simple sum reduction ----------
  // conclusion: 10000000 elements results in 0.28ms kernel time and 138 GB/s throughput
  //reductionBenchmarkSum<int>(atoi(argv[1]));

  //------------ hessian reduction ----------
  //reductionBenchmarkHessian(atoi(argv[1]));

  cudaCheckError();
  return 0;
}
