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

int main() {
  float* data = new float[12];
  //imp::cu::Transformationf myTransform(data);
  //imp::cu::Matrix<imp::Vec32fC1,4,5> myTransform;
  //imp::cu::Matrix3f test;
  imp::cu::Transformationf myTransform(data);
  std::cout << "Rows/Cols "<< myTransform.rows() << "/" << myTransform.cols() << std::endl;
  //float3 myFloat = make_float3(3.0,2.0,1.0);
  //float3 myfloat;
  //myfloat.x = 3;
  //myfloat.y = 4;
  //std::cout << myfloat.x << std::endl;
  cudaCheckError();
  return 0;
}
