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
#include <vikit/math_utils.h>

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

__device__ int kKernelIterations = 1;
int kKernelIterationsCpu = 1;

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

__host__ __device__ __forceinline__ void setGx(imp::cu::Matrix<float,3,6>& __restrict__ g_x,const float3& __restrict__ p_in_imu)
{
  g_x(0,0) = 1.0;
  g_x(0,1) = 0.0;
  g_x(0,2) = 0.0;
  g_x(0,3) = 0.0;
  g_x(0,4) = p_in_imu.z;
  g_x(0,5) = -p_in_imu.y;
  g_x(1,0) = 0.0;
  g_x(1,1) = 1.0;
  g_x(1,2) = 0.0;
  g_x(1,3) = -p_in_imu.z;
  g_x(1,4) = 0.0;
  g_x(1,5) = p_in_imu.x;
  g_x(2,0) = 0.0;
  g_x(2,1) = 0.0;
  g_x(2,2) = 1.0;
  g_x(2,3) = p_in_imu.y;
  g_x(2,4) = -p_in_imu.x;
  g_x(2,5) = 0.0;
}

//Todo: This function should be a member function of the CPU camera
__host__ __device__ __forceinline__ void setPinholeJacobian(imp::cu::Matrix<float,2,3>& __restrict__ jac_cam,
                                                            const float3& __restrict__ p_in_cam, const float& __restrict__ focal_length)
{
  float ratio_p_x_z_cam = p_in_cam.x/p_in_cam.z;
  float ratio_p_y_z_cam = p_in_cam.y/p_in_cam.z;
  float ratio_fl_p_z_cam = focal_length/p_in_cam.z;

  jac_cam(0,0) = ratio_fl_p_z_cam;
  jac_cam(0,1) = 0.0;
  jac_cam(0,2) = -ratio_fl_p_z_cam*ratio_p_x_z_cam;
  jac_cam(1,0) = 0.0;
  jac_cam(1,1) = ratio_fl_p_z_cam;
  jac_cam(1,2) = -ratio_fl_p_z_cam*ratio_p_y_z_cam;
}


__global__ void kernelBaseCachesGeneric(const imp::cu::Matrix<float,3,4> T_imu_cam,const imp::cu::Matrix<float,3,3> R_imu_cam,const float focal_length,
                                        const float3* __restrict__  p_in_cam, float* __restrict__ jac_proj_cache,int nrElements)
{

  int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i < nrElements)
  {
    const float3 p_in_imu = transform(T_imu_cam,p_in_cam[i]);
    imp::cu::Matrix<float,3,6> g_x;
    setGx(g_x,p_in_imu);

    imp::cu::Matrix<float,2,3> jac_cam;
    setPinholeJacobian(jac_cam,p_in_cam[i],focal_length);

    imp::cu::Matrix<float,2,6> jac_proj = ((jac_cam*R_imu_cam)*g_x);

    // wite to buffer
    int offset = 2*6*i;
#pragma unroll
    for(int row = 0; row < 2;++row)
    {
#pragma unroll
      for(int col = 0; col < 6; ++col)
      {
        jac_proj_cache[offset + col] = -1.0*jac_proj(row,col); // times (-1) because of our definition of the photometric error
      }
      offset +=6;
    }
  }
}


__global__ void kernelBaseCachesPinhole(const imp::cu::Matrix<float,3,4> T_imu_cam, const imp::cu::Matrix<float,3,3> R_cam_imu, const float focal_length,
                                        const float3* __restrict__  p_in_cam, float* __restrict__ jac_proj_cache, int nrElements)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < nrElements)
  {
    float3 p_in_imu = transform(T_imu_cam,p_in_cam[i]);
    float ratio_p_x_z_cam = p_in_cam[i].x/p_in_cam[i].z;
    float ratio_p_y_z_cam = p_in_cam[i].y/p_in_cam[i].z;
    float ratio_fl_p_z_cam = (-1.0)*focal_length/p_in_cam[i].z; // times (-1) because of our definition of the photometric error

    float r00 = ratio_fl_p_z_cam*(R_cam_imu(0,0) - R_cam_imu(2,0)*ratio_p_x_z_cam);
    float r01 = ratio_fl_p_z_cam*(R_cam_imu(0,1) - R_cam_imu(2,1)*ratio_p_x_z_cam);
    float r02 = ratio_fl_p_z_cam*(R_cam_imu(0,2) - R_cam_imu(2,2)*ratio_p_x_z_cam);
    float r10 = ratio_fl_p_z_cam*(R_cam_imu(1,0) - R_cam_imu(2,0)*ratio_p_y_z_cam);
    float r11 = ratio_fl_p_z_cam*(R_cam_imu(1,1) - R_cam_imu(2,1)*ratio_p_y_z_cam);
    float r12 = ratio_fl_p_z_cam*(R_cam_imu(1,2) - R_cam_imu(2,2)*ratio_p_y_z_cam);

    int offset = 2*6*i;
    jac_proj_cache[offset] = r00;
    jac_proj_cache[offset + 1] = r01;
    jac_proj_cache[offset + 2] = r02;
    jac_proj_cache[offset + 3] = -p_in_imu.z*r01 + p_in_imu.y*r02;
    jac_proj_cache[offset + 4] = p_in_imu.z*r00 - p_in_imu.x*r02;
    jac_proj_cache[offset + 5] = -p_in_imu.y*r00 + p_in_imu.x*r01;
    jac_proj_cache[offset + 6] = r10;
    jac_proj_cache[offset + 7] = r11;
    jac_proj_cache[offset + 8] = r12;
    jac_proj_cache[offset + 9] = -p_in_imu.z*r11 + p_in_imu.y*r12;
    jac_proj_cache[offset + 10] = p_in_imu.z*r10 - p_in_imu.x*r12;
    jac_proj_cache[offset + 11] = -p_in_imu.y*r10 + p_in_imu.x*r11;
  }
}

void cpuBaseCachesPinhole(const imp::cu::Matrix<float,3,4> T_imu_cam, const imp::cu::Matrix<float,3,3> R_cam_imu, const float focal_length,
                          const float3* __restrict__  p_in_cam, float* __restrict__ jac_proj_cache, int nrElements)
{
  for(int i=0;i<nrElements;++i)
  {
    float3 p_in_imu = transform(T_imu_cam,p_in_cam[i]);
    float ratio_p_x_z_cam = p_in_cam[i].x/p_in_cam[i].z;
    float ratio_p_y_z_cam = p_in_cam[i].y/p_in_cam[i].z;
    float ratio_fl_p_z_cam = (-1.0)*focal_length/p_in_cam[i].z; // times (-1) because of our definition of the photometric error

    float r00 = ratio_fl_p_z_cam*(R_cam_imu(0,0) - R_cam_imu(2,0)*ratio_p_x_z_cam);
    float r01 = ratio_fl_p_z_cam*(R_cam_imu(0,1) - R_cam_imu(2,1)*ratio_p_x_z_cam);
    float r02 = ratio_fl_p_z_cam*(R_cam_imu(0,2) - R_cam_imu(2,2)*ratio_p_x_z_cam);
    float r10 = ratio_fl_p_z_cam*(R_cam_imu(1,0) - R_cam_imu(2,0)*ratio_p_y_z_cam);
    float r11 = ratio_fl_p_z_cam*(R_cam_imu(1,1) - R_cam_imu(2,1)*ratio_p_y_z_cam);
    float r12 = ratio_fl_p_z_cam*(R_cam_imu(1,2) - R_cam_imu(2,2)*ratio_p_y_z_cam);

    int offset = 2*6*i;
    jac_proj_cache[offset] = r00;
    jac_proj_cache[offset + 1] = r01;
    jac_proj_cache[offset + 2] = r02;
    jac_proj_cache[offset + 3] = -p_in_imu.z*r01 + p_in_imu.y*r02;
    jac_proj_cache[offset + 4] = p_in_imu.z*r00 - p_in_imu.x*r02;
    jac_proj_cache[offset + 5] = -p_in_imu.y*r00 + p_in_imu.x*r01;
    jac_proj_cache[offset + 6] = r10;
    jac_proj_cache[offset + 7] = r11;
    jac_proj_cache[offset + 8] = r12;
    jac_proj_cache[offset + 9] = -p_in_imu.z*r11 + p_in_imu.y*r12;
    jac_proj_cache[offset + 10] = p_in_imu.z*r10 - p_in_imu.x*r12;
    jac_proj_cache[offset + 11] = -p_in_imu.y*r10 + p_in_imu.x*r11;
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

// -------------------- test splitting---------------------------------------------------------------
void experiment3()
{
  size_t numFeatures = 20;
  //size_t nrKernelCalls = 1000;
  imp::cu::Matrix<float,3,4> T_imu_cam;
  T_imu_cam(0,0) = 1.0;
  T_imu_cam(1,0) = 0.0;
  T_imu_cam(2,0) = 0.0;
  T_imu_cam(0,1) = 0.0;
  T_imu_cam(1,1) = 1.0;
  T_imu_cam(2,1) = 0.0;
  T_imu_cam(0,2) = 0.0;
  T_imu_cam(1,2) = 0.0;
  T_imu_cam(2,2) = 1.0;
  T_imu_cam(0,3) = 1.0;
  T_imu_cam(1,3) = 2.0;
  T_imu_cam(2,3) = 3.0;

  imp::cu::Matrix<float,3,3> R_cam_imu;
  R_cam_imu(0,0) = 1.0;
  R_cam_imu(1,0) = 0.0;
  R_cam_imu(2,0) = 0.0;
  R_cam_imu(0,1) = 0.0;
  R_cam_imu(1,1) = 1.0;
  R_cam_imu(2,1) = 0.0;
  R_cam_imu(0,2) = 0.0;
  R_cam_imu(1,2) = 0.0;
  R_cam_imu(2,2) = 1.0;

  float focal_length = 125.0;
  float3 p_in_cam_host[numFeatures];
  for(size_t ii=0;ii<numFeatures;ii++)
  {
    p_in_cam_host[ii] = make_float3(ii%2,ii%3,ii%4+1);
    //std::cout << p_in_cam_host[ii].x << " " << p_in_cam_host[ii].y << " " << p_in_cam_host[ii].z << std::endl;
  }

  float3* p_in_cam_dev;
  float* jacobian_dev_pinhole;
  float* jacobian_dev_generic;
  float jacobian_host_pinhole[12*numFeatures];
  float jacobian_host_generic[12*numFeatures];

  cudaMalloc((void**)& p_in_cam_dev,numFeatures*sizeof(float3));
  cudaMalloc((void**)& jacobian_dev_pinhole,numFeatures*12*sizeof(float));
  cudaMalloc((void**)& jacobian_dev_generic,numFeatures*12*sizeof(float));
  cudaMemcpy(p_in_cam_dev,p_in_cam_host,numFeatures*sizeof(float3),cudaMemcpyHostToDevice);


  // split processing into 2 kernel calls
  size_t nr_first_half = numFeatures/2 - 6;
  size_t nr_second_half = numFeatures/2 + 6;
  dim3 threads_jacobian(32);
  dim3 blocks_jacobian((numFeatures+threads_jacobian.x-1)/threads_jacobian.x);
  dim3 blocks_jacobian_first((nr_first_half+threads_jacobian.x-1)/threads_jacobian.x);
  dim3 blocks_jacobian_second((nr_second_half+threads_jacobian.x-1)/threads_jacobian.x);

  kernelBaseCachesPinhole<<<blocks_jacobian_first,threads_jacobian>>>(T_imu_cam,R_cam_imu,focal_length,
                                                                p_in_cam_dev,jacobian_dev_pinhole,nr_first_half);
  cudaDeviceSynchronize();
  // second half
  kernelBaseCachesPinhole<<<blocks_jacobian_second,threads_jacobian>>>(T_imu_cam,R_cam_imu,focal_length,
                                                                &p_in_cam_dev[nr_first_half],&jacobian_dev_pinhole[12*nr_first_half],nr_second_half);
  cudaDeviceSynchronize();


  kernelBaseCachesGeneric<<<blocks_jacobian,threads_jacobian>>>(T_imu_cam,R_cam_imu,focal_length,
                                                                p_in_cam_dev,jacobian_dev_generic,numFeatures);
  cudaDeviceSynchronize();


  //retrieve result
  cudaMemcpy(jacobian_host_pinhole,jacobian_dev_pinhole,numFeatures*12*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(jacobian_host_generic,jacobian_dev_generic,numFeatures*12*sizeof(float),cudaMemcpyDeviceToHost);

  //Verify results
  bool result_jacobian_ok = true;
  for(size_t ii=0; ii < 12*numFeatures;++ii)
  {
    //std::cout << jacobian_host_generic[ii] << "    " << jacobian_host_pinhole[ii] <<std::endl;
    if((jacobian_host_generic[ii]-jacobian_host_pinhole[ii]) > 0.0001||isnan(jacobian_host_generic[ii]-jacobian_host_pinhole[ii]))
    {
      std::cout << ii/12 << std::endl;
      result_jacobian_ok = false;
    }
  }

  std::cout << std::endl;
  std::cout << "--------Result Experiment 3-----------" << std::endl;

  if(!result_jacobian_ok)
    std::cout << "Error:  " << __FUNCTION__ << std::endl;
  else
    std::cout << "Success: " << __FUNCTION__ << std::endl;

  std::cout<< std::endl;
  std::cout<< std::endl;

  cudaFree(p_in_cam_dev);
  cudaFree(jacobian_dev_pinhole);
  cudaFree(jacobian_dev_generic);
  cudaCheckError();
}

// -------------------- test base cache---------------------------------------------------------------
void experiment1()
{

  size_t numFeatures = 480*640/1000;
  size_t nrKernelCalls = 1000;
  imp::cu::Matrix<float,3,4> T_imu_cam;
  T_imu_cam(0,0) = 1.0;
  T_imu_cam(1,0) = 0.0;
  T_imu_cam(2,0) = 0.0;
  T_imu_cam(0,1) = 0.0;
  T_imu_cam(1,1) = 1.0;
  T_imu_cam(2,1) = 0.0;
  T_imu_cam(0,2) = 0.0;
  T_imu_cam(1,2) = 0.0;
  T_imu_cam(2,2) = 1.0;
  T_imu_cam(0,3) = 1.0;
  T_imu_cam(1,3) = 2.0;
  T_imu_cam(2,3) = 3.0;

  imp::cu::Matrix<float,3,3> R_cam_imu;
  R_cam_imu(0,0) = 1.0;
  R_cam_imu(1,0) = 0.0;
  R_cam_imu(2,0) = 0.0;
  R_cam_imu(0,1) = 0.0;
  R_cam_imu(1,1) = 1.0;
  R_cam_imu(2,1) = 0.0;
  R_cam_imu(0,2) = 0.0;
  R_cam_imu(1,2) = 0.0;
  R_cam_imu(2,2) = 1.0;

  float focal_length = 125.0;
  float3 p_in_cam_host[numFeatures];
  for(size_t ii=0;ii<numFeatures;ii++)
  {
    p_in_cam_host[ii] = make_float3(ii%2,ii%3,ii%4+1);
    //std::cout << p_in_cam_host[ii].x << " " << p_in_cam_host[ii].y << " " << p_in_cam_host[ii].z << std::endl;
  }

  float3* p_in_cam_dev;
  float* jacobian_dev_pinhole;
  float* jacobian_dev_generic;
  float jacobian_host_pinhole[12*numFeatures];
  float jacobian_host_generic[12*numFeatures];
  float jacobian_host_cpu[12*numFeatures];

  cudaMalloc((void**)& p_in_cam_dev,numFeatures*sizeof(float3));
  cudaMalloc((void**)& jacobian_dev_pinhole,numFeatures*12*sizeof(float));
  cudaMalloc((void**)& jacobian_dev_generic,numFeatures*12*sizeof(float));
  cudaMemcpy(p_in_cam_dev,p_in_cam_host,numFeatures*sizeof(float3),cudaMemcpyHostToDevice);

  dim3 threads_jacobian(32);
  dim3 blocks_jacobian((numFeatures+threads_jacobian.x-1)/threads_jacobian.x);

  // not evaluated kernel to initialize cuda
  kernelBaseCachesPinhole<<<blocks_jacobian,threads_jacobian>>>(T_imu_cam,R_cam_imu,focal_length,
                                                                p_in_cam_dev,jacobian_dev_pinhole,numFeatures);
  cudaDeviceSynchronize();

  std::clock_t c_start_pinhole = std::clock();
  for(size_t tt = 0; tt < nrKernelCalls;tt++)
  {
    kernelBaseCachesPinhole<<<blocks_jacobian,threads_jacobian>>>(T_imu_cam,R_cam_imu,focal_length,
                                                                  p_in_cam_dev,jacobian_dev_pinhole,numFeatures);
    cudaDeviceSynchronize();
  }
  std::clock_t c_end_pinhole = std::clock();
  double time_pinhole = CLOCK_TO_MS(c_start_pinhole,c_end_pinhole);

  std::clock_t c_start_generic = std::clock();
  for(size_t tt = 0; tt < nrKernelCalls;tt++)
  {
    kernelBaseCachesGeneric<<<blocks_jacobian,threads_jacobian>>>(T_imu_cam,R_cam_imu,focal_length,
                                                                  p_in_cam_dev,jacobian_dev_generic,numFeatures);
    cudaDeviceSynchronize();
  }
  std::clock_t c_end_generic = std::clock();
  double time_generic = CLOCK_TO_MS(c_start_generic,c_end_generic);

  std::clock_t c_start_cpu = std::clock();
  for(size_t tt = 0; tt < nrKernelCalls;tt++)
  {
    cpuBaseCachesPinhole(T_imu_cam,R_cam_imu,focal_length,p_in_cam_host,jacobian_host_cpu,numFeatures);
  }
  std::clock_t c_end_cpu = std::clock();
  double time_cpu = CLOCK_TO_MS(c_start_cpu,c_end_cpu);



  //retrieve result
  cudaMemcpy(jacobian_host_pinhole,jacobian_dev_pinhole,numFeatures*12*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(jacobian_host_generic,jacobian_dev_generic,numFeatures*12*sizeof(float),cudaMemcpyDeviceToHost);

  //Verify results
  bool result_jacobian_ok = true;
  for(size_t ii=0; ii < 12*numFeatures;++ii)
  {
    //std::cout << jacobian_host_generic[ii] << "    " << jacobian_host_pinhole[ii] <<std::endl;
    if((jacobian_host_generic[ii]-jacobian_host_pinhole[ii]) > 0.0001||isnan(jacobian_host_generic[ii]-jacobian_host_pinhole[ii]))
    {
      result_jacobian_ok = false;
    }
  }

  std::cout << std::endl;
  std::cout << "--------Result Experiment 1-----------" << std::endl;
  std::cout << "Nr Features = " << numFeatures << std::endl;
  std::cout << "Time Pinhole " << time_pinhole << std::endl;
  std::cout << "Time Generic " << time_generic << std::endl;
  std::cout << "Time CPU " << time_cpu << std::endl;

  if(!result_jacobian_ok)
    std::cout << "Error:  " << __FUNCTION__ << std::endl;
  else
    std::cout << "Success: " << __FUNCTION__ << std::endl;

  std::cout<< std::endl;
  std::cout<< std::endl;

  cudaFree(p_in_cam_dev);
  cudaFree(jacobian_dev_pinhole);
  cudaFree(jacobian_dev_generic);
  cudaCheckError();
}


// --------------------test const matrix---------------------------------------------------------------
void experiment2()
{
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

  cudaMalloc((void**)& data_dev,kNumElements*sizeof(float3));
  cudaMalloc((void**)& data_transformed_dev,kNumElements*sizeof(float3));
  cudaMemcpy(data_dev,data_host,kNumElements*sizeof(float3),cudaMemcpyHostToDevice);

  dim3 threads_transform(32);
  dim3 blocks_transform((kNumElements+threads_transform.x-1)/threads_transform.x);


  //first kernel execution initialize gpu
  global_memory_kernel_multi<<<blocks_transform,threads_transform>>>(dynamic_transform,data_dev,data_transformed_dev,kNumElements);

  //test global memory
  std::clock_t c_start_gpu_global = std::clock();
  for(size_t ii=0;ii<kNumKernelCalls;ii++)
  {
    global_memory_kernel_multi<<<blocks_transform,threads_transform>>>(dynamic_transform,data_dev,data_transformed_dev,kNumElements);
    cudaDeviceSynchronize();
  }
  std::clock_t c_end_gpu_global = std::clock();
  double time_global = CLOCK_TO_MS(c_start_gpu_global,c_end_gpu_global);

  //test static memory
  std::clock_t c_start_gpu_const = std::clock();
  for(size_t ii=0;ii<kNumKernelCalls;ii++)
  {
    const_memory_kernel_multi<<<blocks_transform,threads_transform>>>(static_transform,data_dev,data_transformed_dev,kNumElements);
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

  std::cout << std::endl;
  std::cout << "--------Result Experiment 2-----------" << std::endl;

  std::cout << "Global   Const   CPU" << std::endl;
  std::cout << std::fixed << std::setprecision(3) << time_global << "   "<< time_const << "   " << time_cpu << std::endl;

  if(result_ok)
    std::cout << "test passed" << std::endl;
  else
    std::cout << "error" << std::endl;

  std::cout << std::endl;
  std::cout << std::endl;

  cudaFree(data_dev);
  cudaFree(data_transformed_dev);
  cudaCheckError();
}


int main() {

  experiment1();
  experiment2();
  experiment3();
  //------- - -  -  -test skew
  Eigen::Vector3d vec_test;
  vec_test(0) = 1; vec_test(1)= 2; vec_test(2) = 3;
  Eigen::Matrix3d skewMatrix = -vk::skew(vec_test);
  std::cout << skewMatrix << std::endl;


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
