#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cuda_runtime.h>
#include <imp/cu_core/cu_const_matrix.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/core/pixel.hpp>
#include <math.h>       /* sqrt */
#include <ctime>
#include <vikit/math_utils.h>

#include <imp/cu_core/cu_texture.cuh>
#include <imp/cu_core/cu_texture2d.cuh>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

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

__global__ void k_jacobianAndRefPatches(imp::cu::Texture2D ref_tex, const float2* __restrict__  uv,const float* __restrict__ jac_proj_cache,
                                        int patch_size, int level,int nrFeatures,
                                        float* __restrict__ jacobian_cache,float* __restrict__ ref_patch_cache)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i < nrFeatures)
  {
    const float upper_left_coord_x = uv[i].x - (patch_size - 1)/2.0f;
    const float upper_left_coord_y = uv[i].y - (patch_size - 1)/2.0f;
    size_t ref_patch_index_offset = patch_size*patch_size*i;
    size_t jacobian_index_offset = patch_size*patch_size*8*i;
    size_t jac_proj_cache_index_offset = 12*i;
    int pixel_counter = 0;

#pragma unroll 4
    for(int rr = 0; rr < patch_size; ++rr)
    {
#pragma unroll 4
      for(int cc = 0; cc < patch_size; ++cc)
      {
        float centerTexel;
        imp::cu::tex2DFetch(centerTexel, ref_tex,upper_left_coord_x + cc, upper_left_coord_y + rr);
        ref_patch_cache[ref_patch_index_offset] = centerTexel;
        float dx_left,dx_right,dy_up,dy_down;
        imp::cu::tex2DFetch(dx_left, ref_tex,upper_left_coord_x + cc - 1, upper_left_coord_y + rr);
        imp::cu::tex2DFetch(dx_right, ref_tex,upper_left_coord_x + cc + 1, upper_left_coord_y + rr);
        imp::cu::tex2DFetch(dy_up, ref_tex,upper_left_coord_x + cc, upper_left_coord_y + rr - 1);
        imp::cu::tex2DFetch(dy_down, ref_tex,upper_left_coord_x + cc, upper_left_coord_y + rr + 1);
        const float dx = 0.5f*(dx_right - dx_left);
        const float dy = 0.5f*(dy_down - dy_up);


#pragma unroll
        for(int ii = 0; ii < 6; ++ii)
        {
          jacobian_cache[jacobian_index_offset + ii] = dx*(jac_proj_cache[jac_proj_cache_index_offset + ii]) + dy*(jac_proj_cache[jac_proj_cache_index_offset + 6 + ii]);
        }
        jacobian_cache[jacobian_index_offset + 6] = -centerTexel;
        jacobian_cache[jacobian_index_offset + 7] = -1;
        jacobian_index_offset += 8;
        ++ref_patch_index_offset;
      }
    }
  }
}

__global__ void k_test_texture(imp::cu::Texture2D ref_tex, float * output, int rows, int cols)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i==0)
  {
    int pixel = 0;
    for(int row = 0; row < rows; ++row)
    {
      for(int col = 0; col < cols; ++col)
      {
        imp::cu::tex2DFetch(output[pixel], ref_tex,col, row);
        ++pixel;
      }
    }
  }
}

void jacobianExperiment()
{
  cv::Mat test(8,8,CV_32FC1);

  test.at<float>(0,0) = 1.0;
  test.at<float>(1,0) = 2.0;
  test.at<float>(0,1) = 3.0;

  std::cout << test << std::endl;

  // generate test data
  cv::Mat testImage = cv::imread("/home/michael/LennaGray.png", CV_LOAD_IMAGE_GRAYSCALE);
  imp::cu::ImageGpu8uC1::Ptr refImage = std::make_shared<imp::cu::ImageGpu8uC1>(imp::ImageCv8uC1(testImage));
  //imp::cu::ImageGpu8uC1 refImage2(imp::ImageCv8uC1(testImage));
  //std::shared_ptr<imp::cu::Texture2D> ref_tex  = std::dynamic_pointer_cast<imp::cu::ImageGpu8uC1>(refImage)
  //    ->genTexture(false,cudaFilterModeLinear,cudaAddressModeWrap,cudaReadModeNormalizedFloat);

  cv::namedWindow("Lenna");
  imp::cu::cvBridgeShow<imp::Pixel8uC1,imp::PixelType::i8uC1>("Lenna",*refImage.get());
  //imp::cu::cvBridgeShow<imp::Pixel8uC1,imp::PixelType::i8uC1>("Lenna",refImage2);

  std::shared_ptr<imp::cu::Texture2D> ref_tex  = refImage->genTexture(false,cudaFilterModeLinear,cudaAddressModeWrap,cudaReadModeNormalizedFloat);
  cv::waitKey(0);

  int rows = 1000;
  int cols = 1000;
  int area = rows*cols;
  float picture[rows*cols];
  float* device_ptr;

  cudaMalloc((void**)&device_ptr,area*sizeof(float));

  k_test_texture<<<1,1>>>(*ref_tex, device_ptr,rows,cols);

  cudaMemcpy(picture,device_ptr,area*sizeof(float),cudaMemcpyDeviceToHost);

  std::cout << "top left   " << static_cast<int>(testImage.at<uint8_t>(0,50)) << std::endl;
  std::cout << "top left 1 " << static_cast<int>(testImage.at<uint8_t>(1,200)) << std::endl;
  std::cout << "top left 2 " << static_cast<int>(testImage.at<uint8_t>(2,100)) << std::endl;

  cv::Mat zoom_float = 255.0f*cv::Mat(rows,cols, CV_32FC1, picture);

  std::cout << "top left zoom   " << static_cast<int>(zoom_float.at<float>(0,50)) << std::endl;
  std::cout << "top left zoom 1 " << static_cast<int>(zoom_float.at<float>(1,200)) << std::endl;
  std::cout << "top left zoom 2 " << static_cast<int>(zoom_float.at<float>(2,100)) << std::endl;

  //std::cout << zoom_float << std::endl;
  cv::Mat zoom_char;
  zoom_float.convertTo(zoom_char, CV_8UC1);
  cv::imshow("Lenna",zoom_char);

  //imp::cu::cvBridgeShow("Lenna",refImage);
  //cv::imshow("Lenna",testImage);
  cv::waitKey(0);
}

__global__ void k_test_interpolation(imp::cu::Texture2D ref_tex, float * output, int patch_size, float u, float v)
{

  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i == 0)
  {
    const float upper_left_coord_x = u- (patch_size - 1)/2.0f;
    const float upper_left_coord_y = v - (patch_size - 1)/2.0f;
    int pixel = 0;
#pragma unroll 4
    for(int rr = 0; rr < patch_size; ++rr)
    {
#pragma unroll 4
      for(int cc = 0; cc < patch_size; ++cc)
      {
        float centerTexel;
        imp::cu::tex2DFetch(centerTexel, ref_tex,upper_left_coord_x + cc, upper_left_coord_y + rr);
        output[pixel] = 255.0f*centerTexel;
        ++pixel;
      }
    }
  }
}

void interpolationTest(float u_, float v_, int patch_size_)
{
  // generate test data
  cv::Mat refImageCv = cv::imread("/home/michael/LennaGray.png", CV_LOAD_IMAGE_GRAYSCALE);
  imp::cu::ImageGpu8uC1::Ptr refImage = std::make_shared<imp::cu::ImageGpu8uC1>(imp::ImageCv8uC1(refImageCv));
  //imp::cu::ImageGpu8uC1 refImage2(imp::ImageCv8uC1(testImage));
  //std::shared_ptr<imp::cu::Texture2D> ref_tex  = std::dynamic_pointer_cast<imp::cu::ImageGpu8uC1>(refImage)
  //    ->genTexture(false,cudaFilterModeLinear,cudaAddressModeWrap,cudaReadModeNormalizedFloat);

  cv::namedWindow("Lenna");
  imp::cu::cvBridgeShow<imp::Pixel8uC1,imp::PixelType::i8uC1>("Lenna",*refImage.get());

  std::shared_ptr<imp::cu::Texture2D> ref_tex  = refImage->genTexture(false,cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat);

  int patch_size = patch_size_;
  float u = u_;
  float v = v_;
  int patch_area = patch_size*patch_size;
  float patch_gpu[patch_area];
  float* device_ptr;

  cudaMalloc((void**)&device_ptr,patch_area*sizeof(float));

  k_test_interpolation<<<1,1>>>(*ref_tex, device_ptr,patch_size,u,v);

  cudaMemcpy(patch_gpu,device_ptr,patch_area*sizeof(float),cudaMemcpyDeviceToHost);

  cv::namedWindow("Patch",cv::WINDOW_NORMAL);
  cv::Mat patch_gpu_cv = cv::Mat(patch_size,patch_size, CV_32FC1, patch_gpu);
  cv::Mat patch_gpu_cv_char;
  patch_gpu_cv.convertTo(patch_gpu_cv_char, CV_8UC1);
  cv::imshow("Patch",patch_gpu_cv_char);
  float patch_cpu[patch_area];

  if(0)
  {
    const int u_ref_i = std::floor(u);
    const int v_ref_i = std::floor(v);

    const int patch_center_ceil = (patch_size - 1)%2 ? (patch_size - 1)/2 + 1 : (patch_size - 1)/2;
    const int stride = refImageCv.step;

    // compute bilateral interpolation weights for reference image
    const float subpix_u_ref = u-u_ref_i;
    const float subpix_v_ref = v-v_ref_i;
    const float wtl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float wtr = subpix_u_ref * (1.0-subpix_v_ref);
    const float wbl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float wbr = subpix_u_ref * subpix_v_ref;

    // interpolate patch with border
    size_t pixel_counter = 0;
    for(int y = 0; y < patch_size; ++y)
    {
      // reference image pointer (openCv stores data in row major format)
      uint8_t* r =
          (uint8_t*) refImageCv.data
          + (v_ref_i-patch_center_ceil+y)*stride
          + (u_ref_i-patch_center_ceil);
      for(int x = 0; x < patch_size; ++x, ++r, ++pixel_counter)
      {
        // precompute interpolated reference patch color
        patch_cpu[pixel_counter] = wtl*r[0] + wtr*r[1] + wbl*r[stride] + wbr*r[stride+1];
      }
    }
  }
  else
  {
    float patch_center = (patch_size - 1)/2.0f;
    float u_tl = u - patch_center;
    float v_tl = v - patch_center;
    const int u_tl_i = std::floor(u_tl);
    const int v_tl_i = std::floor(v_tl);
    const int stride = refImageCv.step;

    // compute bilateral interpolation weights for reference image
    const float subpix_u_tl = u_tl-u_tl_i;
    const float subpix_v_tl = v_tl-v_tl_i;
    const float wtl = (1.0-subpix_u_tl) * (1.0-subpix_v_tl);
    const float wtr = subpix_u_tl * (1.0-subpix_v_tl);
    const float wbl = (1.0-subpix_u_tl) * subpix_v_tl;
    const float wbr = subpix_u_tl * subpix_v_tl;

    // interpolate patch with border
    size_t pixel_counter = 0;
    for(int y = 0; y < patch_size; ++y)
    {
      // reference image pointer (openCv stores data in row major format)
      uint8_t* r =
          (uint8_t*) refImageCv.data
          + (v_tl_i+y)*stride
          + (u_tl_i);
      for(int x = 0; x < patch_size; ++x, ++r, ++pixel_counter)
      {
        // precompute interpolated reference patch color
        patch_cpu[pixel_counter] = wtl*r[0] + wtr*r[1] + wbl*r[stride] + wbr*r[stride+1];
      }
    }
  }


  cv::namedWindow("Patch cpu",cv::WINDOW_NORMAL);
  cv::Mat patch_cpu_cv = cv::Mat(patch_size,patch_size, CV_32FC1, patch_cpu);
  cv::Mat patch_cpu_cv_char;
  patch_cpu_cv.convertTo(patch_cpu_cv_char, CV_8UC1);
  cv::imshow("Patch cpu",patch_cpu_cv_char);

  cv::Mat difference = cv::Mat(patch_size,patch_size, CV_32FC1);
  difference = patch_gpu_cv - patch_cpu_cv;

  double min, max;
  cv::minMaxLoc(difference, &min, &max);
  std::cout << std::endl << std::endl;
  std::cout << "patch gpu" << std::endl;
  std::cout << patch_gpu_cv << std::endl;
  std::cout << "patch cpu" << std::endl;
  std::cout << patch_cpu_cv << std::endl;
  std::cout << "difference" << std::endl;
  std::cout << difference << std::endl;

  std::cout << "Min difference " << min << " Max difference " << max << std::endl;

  cv::waitKey(0);
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
  //size_t numFeatures = 480*640/300;
  // Output in release build:
  //Nr Features = 1024
  //Time Pinhole 12.882
  //Time Generic 15.656
  //Time CPU 12.382
  //Success: experiment1

  size_t numFeatures = 480*640/300;
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

//#pragma GCC push_options
//#pragma GCC optimize ("O0")
//void testBlockOperations()
//{
//  imp::cu::Matrix<float,3,4> mat34;
//  mat34(0,0) = 1.0;
//  mat34(1,0) = 2.0;
//  mat34(2,0) = 3.0;
//  mat34(0,1) = 4.0;
//  mat34(1,1) = 5.0;
//  mat34(2,1) = 6.0;
//  mat34(0,2) = 7.0;
//  mat34(1,2) = 8.0;
//  mat34(2,2) = 9.0;
//  mat34(0,3) = 10.0;
//  mat34(1,3) = 11.0;
//  mat34(2,3) = 12.0;

//  //  template<size_t block_rows,size_t block_cols>
//  //  __host__ __device__ __forceinline__
//  //  Matrix<Type,block_rows,block_cols> blockLoop(size_t top_left_row, size_t top_left_col)
//  //  {
//  //    Matrix<Type,block_rows,block_cols> out;
//  //    for(size_t rr = 0; rr < block_rows; ++rr)
//  //    {
//  //      for(size_t cc = 0; cc < block_cols; ++cc)
//  //      {
//  //        out(rr,cc) = (*this)(rr+top_left_row,cc+top_left_col);
//  //      }
//  //    }
//  //    return out;
//  //  }
//  imp::cu::Matrix<float,3,4> mat33;
//  std::clock_t c_start_block= std::clock();

//  for(int ii=0; ii< 10000; ++ii)
//  {
//    mat33 = mat34.block<3,4>(0,0);
//  }

//  std::clock_t c_end_block = std::clock();
//  double time_block = CLOCK_TO_MS(c_start_block,c_end_block);
//  std::clock_t c_start_block_memcpy= std::clock();
//  for(int ii=0; ii< 10000; ++ii)
//  {
//    mat33 = mat34.block<3,4>(0,0);
//  }
//  std::clock_t c_end_block_memcpy = std::clock();
//  double time_block_memcpy = CLOCK_TO_MS(c_start_block_memcpy,c_end_block_memcpy);
//  std::cout << "Time block " << time_block << std::endl;
//  std::cout << "Time block memcpy " << time_block_memcpy << std::endl;
//  std::cout << mat34 << std::endl;
//  std::cout << mat33 << std::endl;
//}
//#pragma GCC pop_options

//----------------------------------------------- reductionExperiment------------------------------

template <size_t _n_elements>
__host__ __device__ __forceinline__ void setToZero(float*  mem)
{
#pragma unroll
  for(int ind = 0; ind < _n_elements; ++ind)
  {
    mem[ind] = 0.0;
  }
}

template <size_t _matrix_size>
__host__ __device__ __forceinline__ void setVVTUpperTriag(float* __restrict__ upper_triag_row_maj,
                                                          const float* __restrict__ vect,
                                                          const float& __restrict__ weight = 1.0)
{
  int index = 0;
#pragma unroll
  for(int row = 0; row < _matrix_size; ++row)
  {
#pragma unroll
    for(int col = row; col < _matrix_size; ++col,++index)
    {
      upper_triag_row_maj[index] = weight*vect[row]*vect[col];
    }
  }
}

template <size_t _matrix_size>
__host__ __device__ __forceinline__ void addVVTUpperTriag(float* __restrict__ upper_triag_row_maj,
                                                          const float* __restrict__ vect,
                                                          const float& __restrict__ weight = 1.0)
{
  int index = 0;
#pragma unroll
  for(int row = 0; row < _matrix_size; ++row)
  {
#pragma unroll
    for(int col = row; col < _matrix_size; ++col,++index)
    {
      upper_triag_row_maj[index] += weight*vect[row]*vect[col];
    }
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__ void addVector(float* __restrict__ sum_vect,
                                                   const float* __restrict__ addend_vect)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    sum_vect[ind] += addend_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__ void addWeightedVector(float* __restrict__ sum_vect,
                                                           const float* __restrict__ addend_vect,
                                                           const float& __restrict__ weight = 1.0)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    sum_vect[ind] += weight*addend_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__ void subWeightedVector(float* __restrict__ sum_vect,
                                                           const float* __restrict__ addend_vect,
                                                           const float& __restrict__ weight = 1.0)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    sum_vect[ind] -= weight*addend_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__ void setWeightedVector(float* __restrict__ dest_vect,
                                                           const float* __restrict__ src_vect,
                                                           const float& __restrict__ weight = 1.0)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    dest_vect[ind] = weight*src_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__ void copyVector(float* __restrict__ dest_vect,
                                                    const float* __restrict__ src_vect)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    dest_vect[ind] = src_vect[ind];
  }
}


static constexpr unsigned int kJacobianSize = 8;
static constexpr unsigned int kHessianTriagN = 36;
static constexpr unsigned int kPatchSize = 4;
static constexpr unsigned int kPatchArea = kPatchSize*kPatchSize;

// _block_size must be power of 2
template <unsigned int _block_size, bool n_is_pow2>
__global__ void k_jacobianReduceHessianGradient(const float* __restrict__ jacobian_cache,
                                                const float* __restrict__ residual_cache,
                                                const char* __restrict__ visibility_cache,
                                                float* __restrict__ gradient_cache,
                                                float* __restrict__ hessian_cache,
                                                const unsigned int n_elements)
{
  //  extern __shared__ float s_hessian_data[];
  //  extern __shared__ float s_gradient_data[];
  __shared__ float s_hessian_data[_block_size*kHessianTriagN];
  __shared__ float s_gradient_data[_block_size*kJacobianSize];
  float jacobian[kJacobianSize];
  float gradient[kJacobianSize];
  float hessian[kHessianTriagN];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*_block_size*2 + threadIdx.x;
  unsigned int gridSize = _block_size*2*gridDim.x;
  unsigned int hessian_index = tid*kHessianTriagN;
  unsigned int gradient_index = tid*kJacobianSize;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread

  //Get first element
  if((!n_is_pow2)&&(i >= n_elements))
  {
    //set memory to zero
    setToZero<kJacobianSize>(jacobian);
    setToZero<kJacobianSize>(gradient);
    setToZero<kHessianTriagN>(hessian);
  }
  else
  {
    unsigned int visib_index = i/kPatchArea;
    float visible = static_cast<float>(visibility_cache[visib_index]);
    float residual = residual_cache[i];
    //TODO: add weighting function
    float weight = visible;// visible*weight_function(residual/weight_scale);

    copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
    setVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
    setWeightedVector<kJacobianSize>(gradient,jacobian, -weight*residual);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 problem size
    if (n_is_pow2 || i + _block_size < n_elements)
    {
      i += _block_size;
      unsigned int visib_index = i/kPatchArea;
      float visible = static_cast<float>(visibility_cache[visib_index]);
      float residual = residual_cache[i];
      //TODO: add weighting function
      float weight = visible;// visible*weight_function(residual/weight_scale);

      copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
      addVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
      subWeightedVector<kJacobianSize>(gradient,jacobian, weight*residual);
    }
    i += (gridSize - _block_size);
  }

  // Add further elements if available
  while (i < n_elements)
  {
    unsigned int visib_index = i/kPatchArea;
    float visible = static_cast<float>(visibility_cache[visib_index]);
    float residual = residual_cache[i];
    //TODO: add weighting function
    float weight = visible;// visible*weight_function(residual/weight_scale);

    copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
    addVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
    subWeightedVector<kJacobianSize>(gradient,jacobian, weight*residual);
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (n_is_pow2 || i + _block_size < n_elements)
    {
      i += _block_size;
      unsigned int visib_index = i/kPatchArea;
      float visible = static_cast<float>(visibility_cache[visib_index]);
      float residual = residual_cache[i];
      //TODO: add weighting function
      float weight = visible;// visible*weight_function(residual/weight_scale);

      copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
      addVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
      subWeightedVector<kJacobianSize>(gradient,jacobian, weight*residual);
    }
    i += (gridSize - _block_size);
  }

  // each thread puts its local sum into shared memory
  copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
  copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  __syncthreads();

  // do reduction in shared mem
  if ((_block_size >= 512) && (tid < 256))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 256)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 256)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();

  if ((_block_size >= 256) &&(tid < 128))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 128)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 128)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();

  if ((_block_size >= 128) && (tid <  64))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 64)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 64)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();


  //#if (__CUDA_ARCH__ >= 300 )
  //  if ( tid < 32 )
  //  {
  //      // Fetch final intermediate sum from 2nd warp
  //      if (blockSize >=  64) mySum += sdata[tid + 32];
  //      // Reduce final warp using shuffle
  //      for (int offset = warpSize/2; offset > 0; offset /= 2)
  //      {
  //          mySum += __shfl_down(mySum, offset);
  //      }
  //  }
  //#else
  // fully unroll reduction within a single warp
  if ((_block_size >=  64) && (tid < 32))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 32)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 32)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=  32) && (tid < 16))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 16)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 16)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=  16) && (tid <  8))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 8)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 8)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=   8) && (tid <  4))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 4)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 4)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();

  if ((_block_size >=   4) && (tid <  2))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 2)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 2)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=   2) && ( tid <  1))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 1)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 1)*kHessianTriagN]);
  }

  //__syncthreads();
  //#endif

  // write result for this block to global mem
  if (tid == 0)
  {
    copyVector<kJacobianSize>(&gradient_cache[blockIdx.x*kJacobianSize],gradient);
    copyVector<kHessianTriagN>(&hessian_cache[blockIdx.x*kHessianTriagN],hessian);
  }
}

// _block_size must be power of 2
template <unsigned int _block_size, bool n_is_pow2>
__global__ void k_reduceHessianGradient(float* __restrict__ gradient_cache,
                                        float* __restrict__ hessian_cache,
                                        float* __restrict__ gradient_cache_out,
                                        float* __restrict__ hessian_cache_out,
                                        const unsigned int n_elements)
{
  __shared__ float s_hessian_data[_block_size*kHessianTriagN];
  __shared__ float s_gradient_data[_block_size*kJacobianSize];
  float gradient[kJacobianSize];
  float hessian[kHessianTriagN];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*_block_size*2 + threadIdx.x;
  unsigned int gridSize = _block_size*2*gridDim.x;
  unsigned int hessian_index = tid*kHessianTriagN;
  unsigned int gradient_index = tid*kJacobianSize;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread

  //Get first element
  if((!n_is_pow2)&&(i >= n_elements))
  {
    //set memory to zero
    setToZero<kJacobianSize>(gradient);
    setToZero<kHessianTriagN>(hessian);
  }
  else
  {
    copyVector<kJacobianSize>(gradient,&gradient_cache[i*kJacobianSize]);
    copyVector<kHessianTriagN>(hessian,&hessian_cache[i*kHessianTriagN]);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 problem size
    if (n_is_pow2 || i + _block_size < n_elements)
    {
      i += _block_size;
      addVector<kJacobianSize>(gradient,&gradient_cache[i*kJacobianSize]);
      addVector<kHessianTriagN>(hessian,&hessian_cache[i*kHessianTriagN]);

    }
    i += (gridSize - _block_size);
  }

  // Add further elements if available
  while (i < n_elements)
  {
    addVector<kJacobianSize>(gradient,&gradient_cache[i*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&hessian_cache[i*kHessianTriagN]);
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (n_is_pow2 || i + _block_size < n_elements)
    {
      i += _block_size;
      addVector<kJacobianSize>(gradient,&gradient_cache[i*kJacobianSize]);
      addVector<kHessianTriagN>(hessian,&hessian_cache[i*kHessianTriagN]);
    }
    i += (gridSize - _block_size);
  }

  // each thread puts its local sum into shared memory
  copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
  copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  __syncthreads();

  // do reduction in shared mem
  if ((_block_size >= 512) && (tid < 256))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 256)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 256)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();

  if ((_block_size >= 256) &&(tid < 128))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 128)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 128)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();

  if ((_block_size >= 128) && (tid <  64))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 64)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 64)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();


  //#if (__CUDA_ARCH__ >= 300 )
  //  if ( tid < 32 )
  //  {
  //      // Fetch final intermediate sum from 2nd warp
  //      if (blockSize >=  64) mySum += sdata[tid + 32];
  //      // Reduce final warp using shuffle
  //      for (int offset = warpSize/2; offset > 0; offset /= 2)
  //      {
  //          mySum += __shfl_down(mySum, offset);
  //      }
  //  }
  //#else
  // fully unroll reduction within a single warp
  if ((_block_size >=  64) && (tid < 32))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 32)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 32)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=  32) && (tid < 16))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 16)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 16)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=  16) && (tid <  8))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 8)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 8)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=   8) && (tid <  4))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 4)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 4)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  __syncthreads();

  if ((_block_size >=   4) && (tid <  2))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 2)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 2)*kHessianTriagN]);
    // store result to shared memory
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  }

  //__syncthreads();

  if ((_block_size >=   2) && ( tid <  1))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 1)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 1)*kHessianTriagN]);
  }

  //__syncthreads();
  //#endif

  // write result for this block to global mem
  if (tid == 0)
  {
    copyVector<kJacobianSize>(&gradient_cache_out[blockIdx.x*kJacobianSize],gradient);
    copyVector<kHessianTriagN>(&hessian_cache_out[blockIdx.x*kHessianTriagN],hessian);
  }
}

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

bool isPow2(unsigned int x)
{
  return ((x&(x-1))==0);
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

void reduceJacobian(int size, int threads, int blocks,
                    float* jacobian_input_device,
                    char* visibility_input_device,
                    float* residual_input_device,
                    float* gradient_output,
                    float* hessian_output)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2(size))
  {
    switch (threads)
    {

    case 512:
      throw std::runtime_error(" 512 threads exceed the 48kB of available shared memory per block!");
      //      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<512,true>, cudaFuncCachePreferShared);
      //      k_jacobianReduceHessianGradient<512, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
      //                                                                          residual_input_device,
      //                                                                          visibility_input_device,
      //                                                                          gradient_output,
      //                                                                          hessian_output,
      //                                                                          size);
      break;

    case 256:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<256,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<256, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 128:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<128,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<128, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 64:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<64,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<64, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case 32:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<32,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<32, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case 16:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<16,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<16, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  8:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<8,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<8, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;

    case  4:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<4,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<4, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;

    case  2:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<2,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<2, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;

    case  1:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<1,true>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<1, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      throw std::runtime_error(" 512 threads exceed the 48kB of available shared memory per block!");
      //      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<512,false>, cudaFuncCachePreferShared);
      //      k_jacobianReduceHessianGradient<512, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
      //                                                                           residual_input_device,
      //                                                                           visibility_input_device,
      //                                                                           gradient_output,
      //                                                                           hessian_output,
      //                                                                           size);
      break;

    case 256:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<256,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<256, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                           residual_input_device,
                                                                           visibility_input_device,
                                                                           gradient_output,
                                                                           hessian_output,
                                                                           size);
      break;

    case 128:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<128,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<128, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                           residual_input_device,
                                                                           visibility_input_device,
                                                                           gradient_output,
                                                                           hessian_output,
                                                                           size);
      break;

    case 64:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<64,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<64, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 32:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<32,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<32, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 16:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<16,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<16, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case  8:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<8,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<8, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  4:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<4,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<4, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  2:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<2,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<2, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  1:
      cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<1,false>, cudaFuncCachePreferL1);
      k_jacobianReduceHessianGradient<1, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;
    }
  }
  cudaDeviceSynchronize();
}

void reduceHessianGradient(int size, int threads, int blocks,
                           float*  gradient_cache,
                           float*  hessian_cache,
                           float*  gradient_cache_out,
                           float*  hessian_cache_out)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2(size))
  {
    switch (threads)
    {

    case 512:
      throw std::runtime_error(" 512 threads exceed the 48kB of available shared memory per block!");
      //      cudaFuncSetCacheConfig (k_reduceHessianGradient<512,true>, cudaFuncCachePreferL1);
      //      k_reduceHessianGradient<512, true><<< dimGrid, dimBlock >>>(gradient_cache,
      //                                                                  hessian_cache,
      //                                                                  gradient_cache_out,
      //                                                                  hessian_cache_out,
      //                                                                  size);
      break;

    case 256:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<256,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<256, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                  hessian_cache,
                                                                  gradient_cache_out,
                                                                  hessian_cache_out,
                                                                  size);
      break;

    case 128:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<128,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<128, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                  hessian_cache,
                                                                  gradient_cache_out,
                                                                  hessian_cache_out,
                                                                  size);
      break;

    case 64:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<64,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<64, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;

    case 32:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<32,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<32, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;

    case 16:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<16,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<16, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;

    case  8:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<8,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<8, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                hessian_cache,
                                                                gradient_cache_out,
                                                                hessian_cache_out,
                                                                size);
      break;

    case  4:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<4,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<4, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                hessian_cache,
                                                                gradient_cache_out,
                                                                hessian_cache_out,
                                                                size);
      break;

    case  2:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<2,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<2, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                hessian_cache,
                                                                gradient_cache_out,
                                                                hessian_cache_out,
                                                                size);
      break;

    case  1:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<1,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<1, true><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                hessian_cache,
                                                                gradient_cache_out,
                                                                hessian_cache_out,
                                                                size);
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      throw std::runtime_error(" 512 threads exceed the 48kB of available shared memory per block!");
      //      cudaFuncSetCacheConfig (k_reduceHessianGradient<1,false>, cudaFuncCachePreferL1);
      //      k_reduceHessianGradient<1, false><<< dimGrid, dimBlock >>>(gradient_cache,
      //                                                                  hessian_cache,
      //                                                                  gradient_cache_out,
      //                                                                  hessian_cache_out,
      //                                                                  size);
      break;

    case 256:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<256,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<256, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                   hessian_cache,
                                                                   gradient_cache_out,
                                                                   hessian_cache_out,
                                                                   size);
      break;

    case 128:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<128,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<128, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                   hessian_cache,
                                                                   gradient_cache_out,
                                                                   hessian_cache_out,
                                                                   size);
      break;

    case 64:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<64,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<64, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                  hessian_cache,
                                                                  gradient_cache_out,
                                                                  hessian_cache_out,
                                                                  size);
      break;

    case 32:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<32,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<32, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                  hessian_cache,
                                                                  gradient_cache_out,
                                                                  hessian_cache_out,
                                                                  size);
      break;

    case 16:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<16,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<16, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                  hessian_cache,
                                                                  gradient_cache_out,
                                                                  hessian_cache_out,
                                                                  size);
      break;

    case  8:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<8,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<8, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;

    case  4:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<4,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<4, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;

    case  2:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<2,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<2, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;

    case  1:
      cudaFuncSetCacheConfig (k_reduceHessianGradient<1,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<1, false><<< dimGrid, dimBlock >>>(gradient_cache,
                                                                 hessian_cache,
                                                                 gradient_cache_out,
                                                                 hessian_cache_out,
                                                                 size);
      break;
    }
  }
  cudaDeviceSynchronize();
}

void reductionExperiment(int _nr_ele)
{
  //int max_threads = 256;
  int max_threads = 256;
  //int max_threads = 64;
  int nr_patches = _nr_ele;
  int nr_elements = nr_patches*kPatchArea;

  int threads = (nr_elements < max_threads*2) ? nextPow2((nr_elements + 1)/ 2) : max_threads;
  int blocks = (nr_elements + (threads * 2 - 1)) / (threads * 2);

  //Blocks such that each thread sums log(n) elements
  int nr_ele_per_thread = std::floor(log2 (static_cast<double>(nr_elements)));
  int blocks_brent = (nr_elements + (threads*nr_ele_per_thread - 1)) / (threads*nr_ele_per_thread);

  std::cout << "nr_elements = " << nr_elements << std::endl;
  std::cout << "threads = " << threads << std::endl;
  std::cout << "nr_ele_per_thread = " << nr_ele_per_thread << std::endl;
  std::cout << "blocks = " << blocks << std::endl;
  std::cout << "blocks_brent = " << blocks_brent << std::endl;
  std::cout << "nr threads " << blocks*threads << std::endl;
  std::cout << "brent nr threads " << blocks_brent*threads << std::endl;
  std::cout << "effective elements per thread " << static_cast<double>(nr_elements)/static_cast<double>(blocks*threads) << std::endl;
  std::cout << "brent effective elements per thread " << static_cast<double>(nr_elements)/static_cast<double>(blocks_brent*threads) << std::endl;

  // Generate Test data
  unsigned int jacobian_input_size = nr_elements*kJacobianSize;
  float* jacobian_input_host = (float*) malloc(jacobian_input_size*sizeof(float));
  unsigned int visibility_input_size = nr_elements/kPatchArea;
  char* visibility_input_host = (char*) malloc(visibility_input_size*sizeof(char));
  unsigned int residual_input_size = nr_elements;
  float* residual_input_host = (float*) malloc(residual_input_size*sizeof(float));

  float visibility_ratio = 0.2;
  srand(115);

  // Initialize test data
  //std::cout << "jacobian data" << std::endl;
  for(unsigned int ii = 0; ii < jacobian_input_size; ++ii)
  {
    float magnitude = 10;
    float rand_init = (static_cast<float>(rand())/RAND_MAX - 0.5)*magnitude;
    jacobian_input_host[ii] = rand_init;
    //std::cout << jacobian_input_host[ii] << " ";
  }
  //std::cout << std::endl;

  //std::cout << "visibility" << std::endl;
  for(unsigned int ii = 0; ii < visibility_input_size; ++ii)
  {
    char rand_init = (static_cast<float>(rand())/RAND_MAX - visibility_ratio) < 0 ? 1 : 0;
    visibility_input_host[ii] = rand_init;
    //std::cout << static_cast<int>(visibility_input_host[ii]) << " ";
  }
  //std::cout << std::endl;

  //std::cout << "residual" << std::endl;
  for(unsigned int ii = 0; ii < residual_input_size; ++ii)
  {
    float magnitude = 2.0;
    float rand_init = (static_cast<float>(rand())/RAND_MAX - 0.5)*magnitude;
    residual_input_host[ii] = rand_init;
    //std::cout << residual_input_host[ii] << " ";
  }
  //std::cout << std::endl;

  float* jacobian_input_device;
  char* visibility_input_device;
  float* residual_input_device;
  float* gradient_output;
  float* hessian_output;
  float* gradient_output_2;
  float* hessian_output_2;

  cudaMalloc((void **)& jacobian_input_device,jacobian_input_size*sizeof(float));
  cudaMalloc((void **)& visibility_input_device,visibility_input_size*sizeof(char));
  cudaMalloc((void **)& residual_input_device,residual_input_size*sizeof(float));
  cudaMalloc((void **)& gradient_output,std::max(blocks,blocks_brent)*kJacobianSize*sizeof(float));
  cudaMalloc((void **)& hessian_output,std::max(blocks,blocks_brent)*kHessianTriagN*sizeof(float));
  cudaMalloc((void **)& gradient_output_2,std::max(blocks,blocks_brent)*kJacobianSize*sizeof(float));
  cudaMalloc((void **)& hessian_output_2,std::max(blocks,blocks_brent)*kHessianTriagN*sizeof(float));

  cudaMemcpy(jacobian_input_device,jacobian_input_host,jacobian_input_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(visibility_input_device,visibility_input_host,visibility_input_size*sizeof(char),cudaMemcpyHostToDevice);
  cudaMemcpy(residual_input_device,residual_input_host,residual_input_size*sizeof(float),cudaMemcpyHostToDevice);

  //  cudaFuncSetCacheConfig (k_jacobianReduceHessianGradient<16,true>, cudaFuncCachePreferShared);
  //  cudaFuncSetCacheConfig (k_reduceHessianGradient<16,true>, cudaFuncCachePreferShared);

  // ----------------------- reduce normal
  int output_cache = 1;
  reduceJacobian(nr_elements,threads,blocks,jacobian_input_device,visibility_input_device,residual_input_device,gradient_output,hessian_output);
  int problem_size = blocks;
  while(blocks > 1)
  {
    threads = (problem_size < max_threads*2) ? nextPow2((problem_size + 1)/ 2) : max_threads;
    blocks = (problem_size + (threads * 2 - 1)) / (threads * 2);
    reduceHessianGradient(problem_size,threads,blocks,gradient_output,hessian_output,gradient_output_2,hessian_output_2);
    output_cache = 2;
    if(blocks > 1)
    {
      problem_size = blocks;
      threads = (problem_size < max_threads*2) ? nextPow2((problem_size + 1)/ 2) : max_threads;
      blocks = (problem_size + (threads * 2 - 1)) / (threads * 2);
      reduceHessianGradient(problem_size,threads,blocks,gradient_output_2,hessian_output_2,gradient_output,hessian_output);
      output_cache = 1;
    }
  }

  //get result
  float hessian_array_host[kHessianTriagN];
  float gradient_array_host[kJacobianSize];
  if(output_cache = 1)
  {
    cudaMemcpy(hessian_array_host,hessian_output,kHessianTriagN*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient_array_host,gradient_output,kJacobianSize*sizeof(float),cudaMemcpyDeviceToHost);
  }
  else
  {
    cudaMemcpy(hessian_array_host,hessian_output_2,kHessianTriagN*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient_array_host,gradient_output_2,kJacobianSize*sizeof(float),cudaMemcpyDeviceToHost);
  }

  // print result
  std::cout << "Hessian Normal" << std::endl;
  float hessian_output_array[kJacobianSize*kJacobianSize];
  for(unsigned int row = 0, index = 0; row < kJacobianSize; ++row)
  {
    for(unsigned int col = row; col < kJacobianSize; ++col,++index)
    {
      hessian_output_array[row*kJacobianSize + col] = hessian_output_array[col*kJacobianSize + row] =  hessian_array_host[index];
    }
    std::cout << std::endl;
  }

  for(unsigned int row = 0; row < kJacobianSize; ++row)
  {
    for(unsigned int col = 0; col < kJacobianSize; ++col)
    {
      std::cout << std::setw( 12 ) << hessian_output_array[row*kJacobianSize + col] << " " ;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Gradient Normal" << std::endl;
  for(unsigned int ii = 0; ii < kJacobianSize; ++ii)
  {
    std::cout << gradient_array_host[ii] << " " ;
  }
  std::cout << std::endl;


  // ----------------------- reduce brent
  output_cache = 1;
  reduceJacobian(nr_elements,threads,blocks_brent,jacobian_input_device,visibility_input_device,residual_input_device,gradient_output,hessian_output);
  problem_size = blocks_brent;
  while(blocks_brent > 1)
  {
    threads = (problem_size < max_threads*2) ? nextPow2((problem_size + 1)/ 2) : max_threads;
    blocks_brent = (problem_size + (threads * 2 - 1)) / (threads * 2);
    reduceHessianGradient(problem_size,threads,blocks_brent,gradient_output,hessian_output,gradient_output_2,hessian_output_2);
    output_cache = 2;
    if(blocks_brent > 1)
    {
      problem_size = blocks_brent;
      threads = (problem_size < max_threads*2) ? nextPow2((problem_size + 1)/ 2) : max_threads;
      blocks_brent = (problem_size + (threads * 2 - 1)) / (threads * 2);
      reduceHessianGradient(problem_size,threads,blocks_brent,gradient_output_2,hessian_output_2,gradient_output,hessian_output);
      output_cache = 1;
    }
  }

  //get result
  if(output_cache = 1)
  {
    cudaMemcpy(hessian_array_host,hessian_output,kHessianTriagN*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient_array_host,gradient_output,kJacobianSize*sizeof(float),cudaMemcpyDeviceToHost);
  }
  else
  {
    cudaMemcpy(hessian_array_host,hessian_output_2,kHessianTriagN*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient_array_host,gradient_output_2,kJacobianSize*sizeof(float),cudaMemcpyDeviceToHost);
  }

  // print result
  std::cout << "Hessian Brent" << std::endl;
  for(unsigned int row = 0, index = 0; row < kJacobianSize; ++row)
  {
    for(unsigned int col = row; col < kJacobianSize; ++col,++index)
    {
      hessian_output_array[row*kJacobianSize + col] = hessian_output_array[col*kJacobianSize + row] =  hessian_array_host[index];
    }
    std::cout << std::endl;
  }

  for(unsigned int row = 0; row < kJacobianSize; ++row)
  {
    for(unsigned int col = 0; col < kJacobianSize; ++col)
    {
      std::cout << std::setw( 12 ) << hessian_output_array[row*kJacobianSize + col] << " " ;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Gradient Brent" << std::endl;
  for(unsigned int ii = 0; ii < kJacobianSize; ++ii)
  {
    std::cout << gradient_array_host[ii] << " " ;
  }
  std::cout << std::endl;

  Eigen::Matrix<float,kJacobianSize,kJacobianSize> H;
  Eigen::Matrix<float,kJacobianSize,1> g;

  reduceJacobianCPU(nr_elements,
                    jacobian_input_host,
                    visibility_input_host,
                    residual_input_host,
                    H,
                    g);

  std::cout << "Hessian CPU" << std::endl;
  std::cout << H << std::endl;
  std::cout << "Gradient CPU" << std::endl;
  std::cout << g.transpose() << std::endl;

  free(jacobian_input_host);
  free(visibility_input_host);
  free(residual_input_host);
  cudaFree(jacobian_input_device);
  cudaFree(visibility_input_device);
  cudaFree(residual_input_device);
  cudaFree(gradient_output);
  cudaFree(hessian_output);
  cudaFree(gradient_output_2);
  cudaFree(hessian_output_2);

  cudaCheckError();
}

int main(int argc, const char* argv[]) {
  reductionExperiment(atoi(argv[1]));
  //interpolationTest(atof(argv[1]),atof(argv[2]),atoi(argv[3]));
  //jacobianExperiment();
  //experiment1();
  //experiment2();
  //experiment3();
  //testBlockOperations();
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
