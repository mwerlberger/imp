#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cuda_runtime.h>
#include <imp/cu_core/cu_const_matrix.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/core/pixel.hpp>
#include <math.h>       /* sqrt */
#include <ctime>

#define cudaCheckError() {                                                                       \
  cudaError_t e=cudaGetLastError();                                                        \
  if(e!=cudaSuccess) {                                                                     \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
  exit(EXIT_FAILURE);                                                                  \
  }                                                                                        \
  }

#define CLOCK_TO_MS(C_START,C_END) 1000.0 * (C_END - C_START) / CLOCKS_PER_SEC

//template<typename Type, typename inputType>
//void testFunction(Eigen::Matrix<inputType,3,3> input)
//{
//  std::cout << "hello" << std::endl;
//  Eigen::Matrix<Type,3,3> test;
//  test = input.cast<Type>();
//}

__device__ int kKernelIterations = 100;
int kKernelIterationsCpu = 100;

float normFloat3(float3 in)
{
  return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}

float3 sub(float3 in1,float3 in2)
{
  return make_float3(in1.x - in2.x,in1.y - in2.y,in1.z - in2.z);
}

__global__ void const_memory_kernel_single(imp::cu::TransformationStatic trans,float3 in,float3* out) {

  *out = transform(trans,in);
}

__global__ void global_memory_kernel_single(imp::cu::TransformationDynamic trans,float3 in,float3* out) {

  *out = transform(trans,in);
}

__global__ void global_memory_kernel_multi(imp::cu::TransformationDynamic trans, float3* in,float3* out, int nrElements) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < nrElements) {
    for(int ii = 0; ii < kKernelIterations;ii++)
    {
      out[i] = transform(trans,in[i]);
    }
  }
}

//__global__ void global_memory_ldg_kernel_multi(imp::cu::TransformationDynamic trans, __restrict__ float3* in,float3* out, int nrElements) {
//  int i = blockIdx.x*blockDim.x+threadIdx.x;
//  if(i < nrElements) {
//    for(int ii = 0; ii < kKernelIterations;ii++)
//    {
//      out[i] = transformLdg(trans,in[i]);
//    }
//  }
//}

__global__ void const_memory_kernel_multi(imp::cu::TransformationStatic trans,float3* in,float3* out, int nrElements) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < nrElements) {
    for(int ii = 0; ii < kKernelIterations;ii++)
    {
      out[i] = transform(trans,in[i]);
    }
  }
}

//to compare
void host_transform_multi(imp::cu::TransformationStatic trans,float3* in,float3* out, int nrElements){
  for(int i = 0; i < nrElements;i++)
  {
    for(int ii = 0; ii < kKernelIterationsCpu;ii++)
    {
      out[i] = transform(trans,in[i]);
    }
  }
}


int main() {
  Eigen::Matrix<float,3,4,Eigen::RowMajor> eigen_transform;
  eigen_transform(0,0) = 1.0;
  eigen_transform(1,0) = 0.0;
  eigen_transform(2,0) = 0.0;
  eigen_transform(0,1) = 0.0;
  eigen_transform(1,1) = 1.0;
  eigen_transform(2,1) = 0.0;
  eigen_transform(0,2) = 0.0;
  eigen_transform(1,2) = 0.0;
  eigen_transform(2,2) = 1.0;
  eigen_transform(0,3) = 1.0;
  eigen_transform(1,3) = 2.0;
  eigen_transform(2,3) = 3.0;

  imp::cu::TransformationStatic static_transform(eigen_transform);
  imp::cu::TransformationMemoryHdlr mem_Hdl(eigen_transform,imp::cu::TransformationMemoryHdlr::MemoryType::DEVICE_MEMORY);
  imp::cu::TransformationDynamic dynamic_transform(mem_Hdl);

  size_t kNumElements = 32*100;
  size_t kNumKernelCalls = 1000;
  float3 data_host[kNumElements];
  float3 data_transformed_host[kNumElements];
  float3 data_transformed_by_device_host[kNumElements];

  //
  for(size_t ii=0;ii<kNumElements;ii++)
  {
    data_host[ii] = make_float3(ii%2,ii%3,ii%4);
    //std::cout << vec_array_host[ii].x << " " << vec_array_host[ii].y << " " << vec_array_host[ii].z << std::endl;
  }

  float3* data_dev;
  float3* data_transformed_dev;

  dim3 threads(32);
  dim3 blocks((kNumElements+threads.x-1)/threads.x);


  cudaMalloc((void**)& data_dev,kNumElements*sizeof(float3));
  cudaMalloc((void**)& data_transformed_dev,kNumElements*sizeof(float3));
  cudaMemcpy(data_dev,data_host,kNumElements*sizeof(float3),cudaMemcpyHostToDevice);

  //first kernel execution initialize gpu
  global_memory_kernel_multi<<<blocks,threads>>>(dynamic_transform,data_dev,data_transformed_dev,kNumElements);

  //test static memory
  std::clock_t c_start_gpu_const = std::clock();
  for(size_t ii=0;ii<kNumKernelCalls;ii++)
  {
    const_memory_kernel_multi<<<blocks,threads>>>(static_transform,data_dev,data_transformed_dev,kNumElements);
    cudaDeviceSynchronize();
  }
  std::clock_t c_end_gpu_const = std::clock();
  double time_const = CLOCK_TO_MS(c_start_gpu_const,c_end_gpu_const);

  //test cpu
  std::clock_t c_start_cpu = std::clock();
  for(size_t ii=0;ii<kNumKernelCalls;ii++)
  {
    host_transform_multi(static_transform,data_host,data_transformed_host,kNumElements);
  }
  std::clock_t c_end_cpu = std::clock();
  double time_cpu = CLOCK_TO_MS(c_start_cpu,c_end_cpu);

  //test global memory
  std::clock_t c_start_gpu_global = std::clock();
  for(size_t ii=0;ii<kNumKernelCalls;ii++)
  {
    global_memory_kernel_multi<<<blocks,threads>>>(dynamic_transform,data_dev,data_transformed_dev,kNumElements);
    cudaDeviceSynchronize();
  }
  std::clock_t c_end_gpu_global = std::clock();
  double time_global = CLOCK_TO_MS(c_start_gpu_global,c_end_gpu_global);


  std::cout << "Global   Const   CPU" << std::endl;
  std::cout << std::fixed << std::setprecision(3) << time_global << "   "<< time_const << "   " << time_cpu << std::endl;

  //compare results
  cudaMemcpy(data_transformed_by_device_host,data_transformed_dev,kNumElements*sizeof(float3),cudaMemcpyDeviceToHost);

  bool result_ok = true;
  for(size_t ii=0;ii<kNumElements;ii++)
  {
    //std::cout << "norm:" << normFloat3(sub(data_transformed_host[ii],data_transformed_by_device_host[ii])) << std::endl;
    if(normFloat3(sub(data_transformed_host[ii],data_transformed_by_device_host[ii])) > 0.0001 || isnan(normFloat3(sub(data_transformed_host[ii],data_transformed_by_device_host[ii]))))
      result_ok = false;
    //std::cout << data_transformed_by_device_host[ii].x << " " << data_transformed_by_device_host[ii].y << " " << data_transformed_by_device_host[ii].z << std::endl;
  }

  if(result_ok)
    std::cout << "test passed" << std::endl;
  else
    std::cout << "error" << std::endl;

  cudaFree(data_dev);
  cudaFree(data_transformed_dev);
  cudaCheckError();
  return 0;
}



#if 0

Eigen::Matrix<float,3,4,Eigen::ColMajor> input;
input <<  1.0,2.0,3.0,4.0,5.0,6.0;
input(0,0) = 1.0;
input(1,0) = 0.0;
input(2,0) = 0.0;
input(0,1) = 0.0;
input(1,1) = 1.0;
input(2,1) = 0.0;
input(0,2) = 0.0;
input(1,2) = 0.0;
input(2,2) = 1.0;
input(0,3) = 1.0;
input(1,3) = 2.0;
input(2,3) = 3.0;

imp::cu::TransformationStatic test_static(input);
imp::cu::TransformationMemoryHdlr mem_Hdl_host(input,imp::cu::TransformationMemoryHdlr::MemoryType::HOST_MEMORY);
//imp::cu::TransformationDynamic test_dynamic_host(mem_Hdl_host);
imp::cu::TransformationMemoryHdlr mem_Hdl_device(input,imp::cu::TransformationMemoryHdlr::MemoryType::DEVICE_MEMORY);
imp::cu::TransformationDynamic test_dynamic_device(mem_Hdl_device);

std::cout <<"Testing ordering" << std::endl;
for(int row = 0;row < input.rows();row++)
{
  for(int col = 0;col < input.cols();col++)
  {
    std::cout << test_static(row,col) << " ";
  }
  std::cout << std::endl;
}

float3 in = make_float3(1.0,1.0,1.0);
//float3 out = transform(test_dynamic_host,in);
//float3 out2 = transform(test_static,in);
float3 out3_host;
float3 out4_host;
float3* out3_dev;
float3* out4_dev;
cudaMalloc((void**)& out3_dev,sizeof(float3));
cudaMalloc((void**)& out4_dev,sizeof(float3));
//std::cout << in.x << " " << in.y << " " << in.z << std::endl <<  input << std::endl << out.x << " " << out.y << " " << out.z << std::endl << out2.x << " " << out2.y << " " << out2.z << std::endl;

const_memory_kernel_single<<< 1, 1 >>>(test_static,in,out3_dev);
cudaMemcpy(&out3_host,out3_dev,sizeof(float3),cudaMemcpyDeviceToHost);
global_memory_kernel_single<<< 1, 1 >>>(test_dynamic_device,in,out4_dev);
cudaMemcpy(&out4_host,out4_dev,sizeof(float3),cudaMemcpyDeviceToHost);
std::cout << "GPU result static "<< out3_host.x << " " << out3_host.y << " " << out3_host.z << std::endl;
std::cout << "GPU result dynamic "<< out4_host.x << " " << out4_host.y << " " << out4_host.z << std::endl;
cudaFree(out3_dev);
cudaFree(out4_dev);

//---- multiple elements-----


size_t kNumElements = 32*1000;
float3 vec_array_host[kNumElements];
float3 vec_result_host[kNumElements];

for(size_t ii=0;ii<kNumElements;ii++)
{
  vec_array_host[ii] = make_float3(ii%2,ii%3,ii%4);
  //std::cout << vec_array_host[ii].x << " " << vec_array_host[ii].y << " " << vec_array_host[ii].z << std::endl;
}

std::vector<float> gpu_times_const;
std::vector<float> gpu_times_const_kernel;
std::vector<float> gpu_times_global;
std::vector<float> gpu_times_global_kernel;
std::vector<float> cpu_times;

float3* vec_array_dev;
float3* vec_result_dev;

dim3 threads(32);
dim3 blocks((kNumElements+threads.x-1)/threads.x);

float3 vec_compare_host[kNumElements];

double time_malloc,time_memcpy,time_memcpy_2, time_free;

for(int tt = 0; tt < 50; tt++)
{
  std::clock_t c_start_gpu_global = std::clock();
  std::clock_t c_start_gpu_global_malloc = std::clock();
  cudaMalloc((void**)& vec_array_dev,kNumElements*sizeof(float3));
  cudaMalloc((void**)& vec_result_dev,kNumElements*sizeof(float3));
  std::clock_t c_end_gpu_global_malloc = std::clock();
  std::clock_t c_start_gpu_global_memcpy = std::clock();
  cudaMemcpy(vec_array_dev,vec_array_host,kNumElements*sizeof(float3),cudaMemcpyHostToDevice);
  std::clock_t c_end_gpu_global_memcpy = std::clock();
  std::clock_t c_start_gpu_global_kernel = std::clock();
  global_memory_kernel_multi<<<blocks,threads>>>(test_dynamic_device,vec_array_dev,vec_result_dev,kNumElements);
  cudaDeviceSynchronize();
  std::clock_t c_end_gpu_global_kernel = std::clock();
  std::clock_t c_start_gpu_global_memcpy_2 = std::clock();
  cudaMemcpy(vec_result_host,vec_result_dev,kNumElements*sizeof(float3),cudaMemcpyDeviceToHost);
  std::clock_t c_end_gpu_global_memcpy_2 = std::clock();
  std::clock_t c_start_gpu_global_free = std::clock();
  cudaFree(vec_array_dev);
  cudaFree(vec_result_dev);
  std::clock_t c_end_gpu_global_free = std::clock();
  std::clock_t c_end_gpu_global = std::clock();
  gpu_times_global.push_back(1000.0 * (c_end_gpu_global-c_start_gpu_global) / CLOCKS_PER_SEC);
  gpu_times_global_kernel.push_back(1000.0 * (c_end_gpu_global_kernel-c_start_gpu_global_kernel) / CLOCKS_PER_SEC);
  time_malloc =1000.0 * (c_end_gpu_global_malloc-c_start_gpu_global_malloc) / CLOCKS_PER_SEC;
  time_memcpy =1000.0 * (c_end_gpu_global_memcpy-c_start_gpu_global_memcpy) / CLOCKS_PER_SEC;
  time_memcpy_2 =1000.0 * (c_end_gpu_global_memcpy_2-c_start_gpu_global_memcpy_2) / CLOCKS_PER_SEC;
  time_free =1000.0 * (c_end_gpu_global_free-c_start_gpu_global_free) / CLOCKS_PER_SEC;

  std::cout << "Total time (Global Memory): " << gpu_times_global.back() << " of which: "<<time_malloc << " / " << time_memcpy << " / " << time_memcpy_2 << " / " << time_free;
  double comput_time = gpu_times_global.back() - (time_malloc + time_memcpy + time_memcpy_2 + time_free);
  std::cout << " Kernel: " << 100.0 * comput_time/gpu_times_global.back() << "%" << std::endl;

  std::clock_t c_start_gpu_const = std::clock();
  cudaMalloc((void**)& vec_array_dev,kNumElements*sizeof(float3));
  cudaMalloc((void**)& vec_result_dev,kNumElements*sizeof(float3));
  cudaMemcpy(vec_array_dev,vec_array_host,kNumElements*sizeof(float3),cudaMemcpyHostToDevice);
  std::clock_t c_start_gpu_const_kernel = std::clock();
  const_memory_kernel_multi<<<blocks,threads>>>(test_static,vec_array_dev,vec_result_dev,kNumElements);
  cudaDeviceSynchronize();
  std::clock_t c_end_gpu_const_kernel = std::clock();
  cudaMemcpy(vec_result_host,vec_result_dev,kNumElements*sizeof(float3),cudaMemcpyDeviceToHost);
  cudaFree(vec_array_dev);
  cudaFree(vec_result_dev);
  std::clock_t c_end_gpu_const = std::clock();
  gpu_times_const.push_back(1000.0 * (c_end_gpu_const-c_start_gpu_const) / CLOCKS_PER_SEC);
  gpu_times_const_kernel.push_back(1000.0 * (c_end_gpu_const_kernel-c_start_gpu_const_kernel) / CLOCKS_PER_SEC);

  //compare
  std::clock_t c_start_cpu = std::clock();
  host_transform_multi(test_static,vec_array_host,vec_compare_host,kNumElements);
  std::clock_t c_end_cpu = std::clock();
  cpu_times.push_back(1000.0 * (c_end_cpu-c_start_cpu) / CLOCKS_PER_SEC);
}

bool result_ok = true;
for(size_t ii=0;ii<kNumElements;ii++)
{
  if(normFloat3(sub(vec_result_host[ii],vec_compare_host[ii])) > 0.0001)
    result_ok = false;
  //std::cout << vec_result_host[ii].x << " " << vec_result_host[ii].y << " " << vec_result_host[ii].z << std::endl;
}

if(result_ok)
std::cout << "test passed" << std::endl;
else
std::cout << "error" << std::endl;

std::cout << "GPU const | GPU global | CPU" << std::endl;
double mean_gpu_const,mean_gpu_global,mean_cpu,mean_const_kernel,mean_global_kernel;
double var_gpu_const,var_gpu_global,var_cpu;
mean_cpu = mean_gpu_const = mean_gpu_global = mean_const_kernel = mean_global_kernel = 0.0;
var_cpu = var_gpu_const = var_gpu_global = 0.0;
for(int ii = 0; ii < static_cast<int>(cpu_times.size());ii++)
{
  mean_cpu += cpu_times.at(ii);
  mean_gpu_const += gpu_times_const.at(ii);
  mean_gpu_global += gpu_times_global.at(ii);
  mean_const_kernel += gpu_times_const_kernel.at(ii);
  mean_global_kernel += gpu_times_global_kernel.at(ii);
  //    std::cout << std::fixed << std::setprecision(3) << gpu_times_const.at(ii) << "      " << gpu_times_global.at(ii) << "       " << cpu_times.at(ii) << std::endl;
}

mean_cpu = mean_cpu/static_cast<double>(cpu_times.size());
mean_gpu_const = mean_gpu_const/static_cast<double>(cpu_times.size());
mean_gpu_global = mean_gpu_global/static_cast<double>(cpu_times.size());
mean_global_kernel = mean_global_kernel/static_cast<double>(cpu_times.size());
mean_const_kernel = mean_const_kernel/static_cast<double>(cpu_times.size());

// compute variance
for(int ii = 0; ii < static_cast<int>(cpu_times.size());ii++)
{
  var_cpu += (cpu_times.at(ii)-mean_cpu)*(cpu_times.at(ii)-mean_cpu);
  var_gpu_const += (gpu_times_const.at(ii) - mean_gpu_const)*(gpu_times_const.at(ii) - mean_gpu_const);
  var_gpu_global += (gpu_times_global.at(ii) - mean_gpu_global)*(gpu_times_global.at(ii) - mean_gpu_global);
}
var_cpu = var_cpu/static_cast<int>(cpu_times.size());
var_gpu_const = var_gpu_const/static_cast<int>(cpu_times.size());
var_gpu_global = var_gpu_global/static_cast<int>(cpu_times.size());


std::cout << "mean" << std::endl;
std::cout << std::fixed << std::setprecision(3) << mean_gpu_const << " (" << mean_const_kernel<< ") "<< "      " << mean_gpu_global << " (" << mean_global_kernel<< ") "<< "       " << mean_cpu << std::endl;
std::cout << "std deviation" << std::endl;
std::cout << std::fixed << std::setprecision(3) << sqrt(var_gpu_const) << "      " << sqrt(var_gpu_global) << "       " << sqrt(var_cpu) << std::endl;

#endif
