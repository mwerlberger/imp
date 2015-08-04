#ifndef __KERNEL_REDUCE_H__
#define __KERNEL_REDUCE_H__

static constexpr unsigned int kJacobianSize = 8;
static constexpr unsigned int kHessianTriagN = 36;
static constexpr unsigned int kPatchSize = 16;
static constexpr unsigned int kPatchArea = kPatchSize*kPatchSize;

template <class T>
void reduce(int size, int threads, int blocks,
            T *d_idata, T *d_odata);

void reduceHessianGradient(const int size, const int threads, const int blocks,
                    const float* __restrict__ jacobian_input_device,
                    const char* __restrict__ visibility_input_device,
                    const float* __restrict__ residual_input_device,
                    float* __restrict__ gradient_output,
                    float* __restrict__ hessian_output);

#endif
