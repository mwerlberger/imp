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

//__global__ void kernel(imp::cu::Transformation trans) {
  //*out = deviceTest.multiply(4);//input->f_;
  //*out = input->multiply(4);//input->f_;
//}

int main() {
  Eigen::Matrix<float,3,4,Eigen::ColMajor> input;
  input <<  1.0,2.0,3.0,4.0,5.0,6.0;
  input(0,0) = 1.0;
  input(1,0) = 2.0;
  input(2,0) = 3.0;
  input(0,1) = 4.0;

  imp::cu::TransformationStatic test(input);
  imp::cu::TransformationMemoryHdlr mem_Hdl(input,imp::cu::TransformationMemoryHdlr::MemoryType::HOST_MEMORY);
  imp::cu::TransformationDynamic test_dynamic(mem_Hdl);

  float3 in = make_float3(1.0,2.0,3.0);
  float3 out = transform(test_dynamic,in);
  std::cout << in.x << " " << in.y << " " << in.z << std::endl <<  input << std::endl << out.x << " " << out.y << " " << out.z << std::endl;

  //kernel<<< 1, 1 >>>(input);
  cudaCheckError();
  return 0;
}
