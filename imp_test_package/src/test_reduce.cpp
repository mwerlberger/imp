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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <Eigen/Dense>
#pragma GCC diagnostic pop

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

void setupInputData(float** jacobian_input, char** visibility_input, float** residual_input, unsigned int nr_elements)
{
  unsigned int jacobian_input_size = nr_elements*kJacobianSize;
  *jacobian_input = (float*) malloc(jacobian_input_size*sizeof(float));
  if(nr_elements%kPatchArea != 0)
  {
    throw std::runtime_error("nr elements must be a multiple of the patchsize");
  }

  unsigned int visibility_input_size = nr_elements/kPatchArea;
  *visibility_input = (char*) malloc(visibility_input_size*sizeof(char));
  unsigned int residual_input_size = nr_elements;
  *residual_input = (float*) malloc(residual_input_size*sizeof(float));

  float visibility_ratio = 0.9;
  srand(115);

  // Initialize test data
  for(unsigned int ii = 0; ii < jacobian_input_size; ++ii)
  {
    float magnitude = 10;
    float rand_init = (static_cast<float>(rand())/RAND_MAX - 0.5)*magnitude;
    (*jacobian_input)[ii] = rand_init;
  }

  for(unsigned int ii = 0; ii < visibility_input_size; ++ii)
  {
    char rand_init = (static_cast<float>(rand())/RAND_MAX - visibility_ratio) < 0 ? 1 : 0;
    (*visibility_input)[ii] = rand_init;
  }

  for(unsigned int ii = 0; ii < residual_input_size; ++ii)
  {
    float magnitude = 2.0;
    float rand_init = (static_cast<float>(rand())/RAND_MAX - 0.5)*magnitude;
    (*residual_input)[ii] = rand_init;
  }
}


void reduceHessianGradientCPU(const int num_blocks,
                              const float* __restrict__ gradient_input_host,
                              const float* __restrict__ hessian_input_host,
                              float gradient_out[kJacobianSize],
                              float hessian_out[kHessianTriagN])
{

  memset(hessian_out,0,kHessianTriagN*sizeof(float));
  memset(gradient_out,0,kJacobianSize*sizeof(float));

  for(unsigned int ii = 0; ii< static_cast<unsigned int>(num_blocks); ++ii)
  {
    for(unsigned int hh = 0; hh < kHessianTriagN; ++hh)
    {
      hessian_out[hh] += hessian_input_host[ii*kHessianTriagN + hh];
    }

    for(unsigned int gg = 0; gg < kJacobianSize; ++gg)
    {
      gradient_out[gg] += gradient_input_host[ii*kJacobianSize + gg];
    }
  }
}

void reductionBenchmarkHessian(int _nr_patches)
{
  unsigned int nr_kernel_calls = 100;
  int max_threads = 256;
  int max_blocks = 64;
  //int nr_elements_per_thread = 2;
  unsigned int nr_patches = static_cast<unsigned int>(_nr_patches);
  unsigned int nr_elements = nr_patches*kPatchArea;
  int nr_elements_per_thread = std::floor(log2 (static_cast<double>(nr_elements)));

  int num_blocks = 0;
  int num_threads = 0;
  getNumBlocksAndThreads(nr_elements, max_blocks, max_threads, nr_elements_per_thread, num_blocks, num_threads);

  // Generate Test data
  float* jacobian_input_host;
  char* visibility_input_host;
  float* residual_input_host;

  setupInputData(&jacobian_input_host,&visibility_input_host, &residual_input_host, nr_elements);

  // host intermediate input
  float* hessian_input_host = (float*) malloc(num_blocks*kHessianTriagN*sizeof(float));
  float* gradient_input_host = (float*) malloc(num_blocks*kJacobianSize*sizeof(float));
  float gradient_out_cpu[kJacobianSize];
  float hessian_out_cpu[kHessianTriagN];

  // Setup device memory
  float* jacobian_input_device;
  char* visibility_input_device;
  float* residual_input_device;
  float* gradient_output;
  float* hessian_output;

  checkCudaErrors(cudaMalloc((void **)& jacobian_input_device,nr_elements*kJacobianSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)& visibility_input_device,(nr_elements/kPatchArea)*sizeof(char)));
  checkCudaErrors(cudaMalloc((void **)& residual_input_device,nr_elements*sizeof(float)));
  checkCudaErrors(cudaMemcpy(jacobian_input_device,jacobian_input_host,nr_elements*kJacobianSize*sizeof(float),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(visibility_input_device,visibility_input_host,(nr_elements/kPatchArea)*sizeof(char),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(residual_input_device,residual_input_host,nr_elements*sizeof(float),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)& gradient_output,num_blocks*kJacobianSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)& hessian_output,num_blocks*kHessianTriagN*sizeof(float)));

  // warm up kernel call
  reduceHessianGradient(nr_elements ,num_threads , num_blocks ,jacobian_input_device ,visibility_input_device ,residual_input_device ,gradient_output ,hessian_output);

  std::clock_t c_start_gpu = std::clock();
  for(unsigned int ii = 0; ii < nr_kernel_calls; ++ii)
  {
    reduceHessianGradient(nr_elements ,num_threads , num_blocks ,jacobian_input_device ,visibility_input_device ,residual_input_device ,gradient_output ,hessian_output);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(hessian_input_host,hessian_output,num_blocks*kHessianTriagN*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(gradient_input_host,gradient_output,num_blocks*kJacobianSize*sizeof(float), cudaMemcpyDeviceToHost));
    reduceHessianGradientCPU(num_blocks,gradient_input_host, hessian_input_host, gradient_out_cpu, hessian_out_cpu);
  }
  std::clock_t c_end_gpu = std::clock();
  double time_gpu_ms = CLOCK_TO_MS(c_start_gpu,c_end_gpu)/((double) nr_kernel_calls);
  unsigned int nr_elements_bytes = nr_elements*kJacobianSize*sizeof(float);
  std::cout << "Time per kernel (ms): " << time_gpu_ms << std::endl << "Troughput (jacobian input data / reduce time):" << (1.0e-9 * ((double) nr_elements_bytes))/(time_gpu_ms/1000) << " GB/s " << std::endl;

  free(jacobian_input_host);
  free(visibility_input_host);
  free(residual_input_host);
  cudaFree(jacobian_input_device);
  cudaFree(visibility_input_device);
  cudaFree(residual_input_device);
  cudaFree(gradient_output);
  cudaFree(hessian_output);
}

void reduceJacobianCPU(int size,
                       float* jacobian_input,
                       char* visibility_input,
                       float* residual_input,
                       Eigen::Matrix<float,kJacobianSize,kJacobianSize>& H,
                       Eigen::Matrix<float,kJacobianSize,1>& g)
{
  H.setZero(kJacobianSize,kJacobianSize);
  g.setZero(kJacobianSize,1);
  for(size_t ii = 0; ii < size/kPatchArea; ++ii)
  {
    if(visibility_input[ii] == 1)
    {
      size_t patch_offset = ii*kPatchArea;
      size_t jacobian_offset = ii*kJacobianSize*kPatchArea;
      for(size_t jj = 0; jj < kPatchArea; ++jj)
      {
        float res = residual_input[patch_offset+jj];

        // Robustification.
        float weight = 1.0;

        // Compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error).
        const Eigen::Map<Eigen::Matrix<float,kJacobianSize,1> > J_d = Eigen::Map<Eigen::Matrix<float,kJacobianSize,1> >(&jacobian_input[jacobian_offset + kJacobianSize*jj]);
        H.noalias() += J_d*J_d.transpose()*weight;
        g.noalias() -= J_d*res*weight;
      }
    }
  }
}

int main(int argc, const char* argv[]) {

  //------------ simple sum reduction ----------
  // conclusion: 10000000 elements results in 0.28ms kernel time and 138 GB/s throughput
  //reductionBenchmarkSum<int>(atoi(argv[1]));

  //------------ hessian reduction ----------
  reductionBenchmarkHessian(atoi(argv[1]));

  cudaCheckError();
  return 0;
}
