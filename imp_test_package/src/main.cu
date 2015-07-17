#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <imp/cu_core/cu_const_matrix.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/core/pixel.hpp>

#define cudaCheckError() {                                                                       \
  cudaError_t e=cudaGetLastError();                                                        \
  if(e!=cudaSuccess) {                                                                     \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
  exit(EXIT_FAILURE);                                                                  \
  }                                                                                        \
  }

//template<typename Type, typename inputType>
//void testFunction(Eigen::Matrix<inputType,3,3> input)
//{
//  std::cout << "hello" << std::endl;
//  Eigen::Matrix<Type,3,3> test;
//  test = input.cast<Type>();
//}

__global__ void const_memory_kernel(imp::cu::TransformationStatic trans,float3 in,float3* out) {

  *out = transform(trans,in);
}

__global__ void global_memory_kernel(imp::cu::TransformationDynamic trans,float3 in,float3* out) {

  *out = transform(trans,in);
}

int main() {
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
  imp::cu::TransformationDynamic test_dynamic_host(mem_Hdl_host);
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
  float3 out = transform(test_dynamic_host,in);
  float3 out2 = transform(test_static,in);
  float3 out3_host;
  float3 out4_host;
  float3* out3_dev;
  float3* out4_dev;
  cudaMalloc((void**)& out3_dev,sizeof(float3));
  cudaMalloc((void**)& out4_dev,sizeof(float3));
  std::cout << in.x << " " << in.y << " " << in.z << std::endl <<  input << std::endl << out.x << " " << out.y << " " << out.z << std::endl << out2.x << " " << out2.y << " " << out2.z << std::endl;

  const_memory_kernel<<< 1, 1 >>>(test_static,in,out3_dev);
  cudaMemcpy(&out3_host,out3_dev,sizeof(float3),cudaMemcpyDeviceToHost);
  global_memory_kernel<<< 1, 1 >>>(test_dynamic_device,in,out4_dev);
  cudaMemcpy(&out4_host,out4_dev,sizeof(float3),cudaMemcpyDeviceToHost);
  std::cout << "GPU result static "<< out3_host.x << " " << out3_host.y << " " << out3_host.z << std::endl;
  std::cout << "GPU result dynamic "<< out4_host.x << " " << out4_host.y << " " << out4_host.z << std::endl;

  cudaCheckError();
  return 0;
}
