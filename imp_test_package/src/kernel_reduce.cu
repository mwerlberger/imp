#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <imp/imp_test_package/kernel_reduce.hpp>

//-------------------------------------------------------------------------------------------
// simple sum reduction
//-------------------------------------------------------------------------------------------

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    __shared__ T sdata[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

extern "C"
bool isPow2(unsigned int x);

template <class T>
void reduce(int size, int threads, int blocks,
            T *d_idata, T *d_odata)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2(size))
  {
    switch (threads)
    {
    case 512:
      reduce6<T, 512, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 256:
      reduce6<T, 256, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 128:
      reduce6<T, 128, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 64:
      reduce6<T,  64, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 32:
      reduce6<T,  32, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 16:
      reduce6<T,  16, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  8:
      reduce6<T,   8, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  4:
      reduce6<T,   4, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  2:
      reduce6<T,   2, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  1:
      reduce6<T,   1, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      reduce6<T, 512, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 256:
      reduce6<T, 256, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 128:
      reduce6<T, 128, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 64:
      reduce6<T,  64, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 32:
      reduce6<T,  32, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case 16:
      reduce6<T,  16, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  8:
      reduce6<T,   8, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  4:
      reduce6<T,   4, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  2:
      reduce6<T,   2, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;

    case  1:
      reduce6<T,   1, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;
    }
  }
}

//-------------------------------------------------------------------------------------------
// Hessian reduction
//-------------------------------------------------------------------------------------------

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

// _block_size must be power of 2
template <unsigned int _block_size, bool n_is_pow2>
__global__ void k_reduceHessianGradient(const float* __restrict__ jacobian_cache,
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

  //__syncthreads();

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


void reduceHessianGradient(const int size, const int threads, const int blocks,
                           const float* __restrict__ jacobian_input_device,
                           const char* __restrict__ visibility_input_device,
                           const float* __restrict__ residual_input_device,
                           float* __restrict__ gradient_output,
                           float* __restrict__ hessian_output)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2(size))
  {
    switch (threads)
    {

    case 512:
      throw std::runtime_error(" 512 threads exceed the 48kB of available shared memory per block!");
      //      cudaFuncSetCacheConfig (k_reduceHessianGradient<512,true>, cudaFuncCachePreferShared);
      //      k_jacobianReduceHessianGradient<512, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
      //                                                                          residual_input_device,
      //                                                                          visibility_input_device,
      //                                                                          gradient_output,
      //                                                                          hessian_output,
      //                                                                          size);
      break;

    case 256:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<256,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<256, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 128:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<128,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<128, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 64:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<64,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<64, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case 32:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<32,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<32, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case 16:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<16,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<16, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  8:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<8,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<8, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;

    case  4:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<4,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<4, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;

    case  2:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<2,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<2, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                        residual_input_device,
                                                                        visibility_input_device,
                                                                        gradient_output,
                                                                        hessian_output,
                                                                        size);
      break;

    case  1:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<1,true>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<1, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
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
      //      cudaFuncSetCacheConfig (k_reduceHessianGradient<512,false>, cudaFuncCachePreferShared);
      //      k_jacobianReduceHessianGradient<512, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
      //                                                                           residual_input_device,
      //                                                                           visibility_input_device,
      //                                                                           gradient_output,
      //                                                                           hessian_output,
      //                                                                           size);
      break;

    case 256:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<256,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<256, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                           residual_input_device,
                                                                           visibility_input_device,
                                                                           gradient_output,
                                                                           hessian_output,
                                                                           size);
      break;

    case 128:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<128,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<128, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                           residual_input_device,
                                                                           visibility_input_device,
                                                                           gradient_output,
                                                                           hessian_output,
                                                                           size);
      break;

    case 64:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<64,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<64, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 32:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<32,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<32, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case 16:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<16,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<16, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                          residual_input_device,
                                                                          visibility_input_device,
                                                                          gradient_output,
                                                                          hessian_output,
                                                                          size);
      break;

    case  8:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<8,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<8, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  4:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<4,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<4, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  2:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<2,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<2, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;

    case  1:
      //cudaFuncSetCacheConfig (k_reduceHessianGradient<1,false>, cudaFuncCachePreferL1);
      k_reduceHessianGradient<1, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                         residual_input_device,
                                                                         visibility_input_device,
                                                                         gradient_output,
                                                                         hessian_output,
                                                                         size);
      break;
    }
  }
}

// Instantiate the reduction function for 3 types
template
void reduce<int>(int size, int threads, int blocks,
            int* d_idata, int* d_odata);

template
void reduce<float>(int size, int threads, int blocks,
            float* d_idata, float* d_odata);


