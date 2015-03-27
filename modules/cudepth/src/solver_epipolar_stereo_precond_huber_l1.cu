#include <imp/cudepth/solver_epipolar_stereo_precond_huber_l1.cuh>

#include <cuda_runtime.h>

#include <glog/logging.h>

#include <imp/cudepth/variational_stereo_parameters.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/cuimgproc/cu_image_filter.cuh>
#include <imp/cuimgproc/cu_image_transform.cuh>
#include <imp/cucore/cu_utils.hpp>
#include <imp/cucore/cu_texture.cuh>
#include <imp/cucore/cu_math.cuh>
#include <imp/cucore/cu_k_setvalue.cuh>

#include "cu_k_warped_gradients.cuh"
#include "cu_k_stereo_ctf_warping_level_precond_huber_l1.cuh"
//#include "k_epipolar_stereo_precond_huber_l1.cu"

namespace imp {
namespace cu {

//------------------------------------------------------------------------------
SolverEpipolarStereoPrecondHuberL1::~SolverEpipolarStereoPrecondHuberL1()
{
  // thanks to smart pointers
}

//------------------------------------------------------------------------------
SolverEpipolarStereoPrecondHuberL1::SolverEpipolarStereoPrecondHuberL1(
    const std::shared_ptr<Parameters>& params, imp::Size2u size, size_type level,
    ConstVectorImagePtr init_correspondence_guess,
    ConstVectorImagePtr init_epi_vec)
  : SolverStereoAbstract(params, size, level)
{
  u_.reset(new Image(size));
  u_prev_.reset(new Image(size));
  u0_.reset(new Image(size));
  pu_.reset(new VectorImage(size));
  q_.reset(new Image(size));
  iw_.reset(new Image(size));
  ix_.reset(new Image(size));
  it_.reset(new Image(size));
  xi_.reset(new Image(size));

  // and its textures
  u_tex_ = u_->genTexture(false, cudaFilterModeLinear);
  u_prev_tex_ =  u_prev_->genTexture(false, cudaFilterModeLinear);
  u0_tex_ =  u0_->genTexture(false, cudaFilterModeLinear);
  pu_tex_ =  pu_->genTexture(false, cudaFilterModeLinear);
  q_tex_ =  q_->genTexture(false, cudaFilterModeLinear);
  ix_tex_ =  ix_->genTexture(false, cudaFilterModeLinear);
  it_tex_ =  it_->genTexture(false, cudaFilterModeLinear);
  xi_tex_ =  xi_->genTexture(false, cudaFilterModeLinear);

  if (init_correspondence_guess && init_epi_vec)
  {
    LOG(INFO) << "SolverEpipolarStereoPrecondHuberL1 created with epipolar constraints"
              << " (" << level_ << ")";
    if (level_ == 0)
    {
      LOG(INFO) << "level 0 -- size: " << size << "(simply setting it)";
      correspondence_guess_ = init_correspondence_guess;
      epi_vec_ = init_epi_vec;
    }
    else
    {
      Fragmentation<16,16> frag(size);

      float downscale_factor = 0.5f*((float)size.width()/(float)init_correspondence_guess->width()+
                                     (float)size.height()/(float)init_correspondence_guess->height());

      LOG(INFO) << "level " << level << "; size: " << size
                << "; downscale_factor:" << downscale_factor;

      correspondence_guess_.reset(new VectorImage(size));
      epi_vec_.reset(new VectorImage(size));

      imp::cu::resample(*correspondence_guess_, *init_correspondence_guess,
                        InterpolationMode::point, false);
      *correspondence_guess_ *= downscale_factor;
//      imp::cu::k_pixelWiseMul
//          <<<
//            frag.dimGrid, frag.dimBlock
//          >>> (correspondence_guess_->data(), correspondence_guess_->stride(),
//               imp::Pixel32fC1(downscale_factor),
//               correspondence_guess_->width(), correspondence_guess_->height());
      imp::cu::resample(*epi_vec_, *init_epi_vec, InterpolationMode::point, false);
      *epi_vec_ *= downscale_factor;
//      imp::cu::k_pixelWiseMul
//          <<<
//            frag.dimGrid, frag.dimBlock
//          >>> (epi_vec_->data(), epi_vec_->stride(),
//               imp::Pixel32fC1(downscale_factor),
//               epi_vec_->width(), epi_vec_->height());
    }
  }
  else
  {
    LOG(WARNING) << "SolverEpipolarStereoPrecondHuberL1 created without epipolar constraints";
  }
}

//------------------------------------------------------------------------------
void SolverEpipolarStereoPrecondHuberL1::init()
{
  u_->setValue(0.0f);
  pu_->setValue(0.0f);
  q_->setValue(0.0f);
  // other variables are init and/or set when needed!
}

//------------------------------------------------------------------------------
void SolverEpipolarStereoPrecondHuberL1::init(const SolverStereoAbstract& rhs)
{
  const SolverEpipolarStereoPrecondHuberL1* from =
      dynamic_cast<const SolverEpipolarStereoPrecondHuberL1*>(&rhs);

  float inv_sf = 1./params_->ctf.scale_factor; // >1 for adapting prolongated disparities

  if(params_->ctf.apply_median_filter)
  {
    imp::cu::filterMedian3x3(*from->u0_, *from->u_);
    imp::cu::resample(*u_, *from->u0_, imp::InterpolationMode::point, false);
  }
  else
  {
    imp::cu::resample(*u_, *from->u_, imp::InterpolationMode::point, false);
  }
  *u_ *= inv_sf;

  imp::cu::resample(*pu_, *from->pu_, imp::InterpolationMode::point, false);
  imp::cu::resample(*q_, *from->q_, imp::InterpolationMode::point, false);

  if (params_->verbose > 2)
  {
    std::cout << "inv_sf: " << inv_sf << std::endl;
    imp::Pixel32fC1 min_val,max_val;
    imp::cu::minMax(u_, min_val, max_val);
    std::cout << "disp: min: " << min_val.x << " max: " << max_val.x << std::endl;
  }
}

//------------------------------------------------------------------------------
void SolverEpipolarStereoPrecondHuberL1::solve(std::vector<ImagePtr> images)
{
  if (params_->verbose > 0)
    std::cout << "SolverEpipolarStereoPrecondHuberL1: solving level " << level_ << " with " << images.size() << " images" << std::endl;

  // sanity check:
  // TODO

  // textures
  i1_tex_ = images.at(0)->genTexture(false, cudaFilterModeLinear);
  i2_tex_ = images.at(1)->genTexture(false, cudaFilterModeLinear);
  correspondence_guess_tex_ =  correspondence_guess_->genTexture(false, cudaFilterModeLinear);
  epi_vec_tex_ =  epi_vec_->genTexture(false, cudaFilterModeLinear);

  u_->copyTo(*u_prev_);
  Fragmentation<16,16> frag(size_);

  // constants
  constexpr float tau = 0.95f;
  constexpr float sigma = 0.95f;
  float lin_step = 0.5f;

  // precond
  constexpr float eta = 2.0f;

  // warping
  for (std::uint32_t warp = 0; warp < params_->ctf.warps; ++warp)
  {
    if (params_->verbose > 5)
      std::cout << "SOLVING warp iteration of Huber-L1 stereo model." << std::endl;

    u_->copyTo(*u0_);

    // compute warped spatial and temporal gradients
    k_warpedGradientsEpipolarConstraint
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (iw_->data(), ix_->data(), it_->data(), ix_->stride(), ix_->width(), ix_->height(),
             *i1_tex_, *i2_tex_, *u0_tex_, *correspondence_guess_tex_, *epi_vec_tex_);

    // compute preconditioner
    k_preconditioner
        <<<
          frag.dimGrid, frag.dimBlock
        >>> (xi_->data(), xi_->stride(), xi_->width(), xi_->height(),
             params_->lambda, *ix_tex_);


    for (std::uint32_t iter = 0; iter < params_->ctf.iters; ++iter)
    {
      // dual update kernel
      k_dualUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (pu_->data(), pu_->stride(), q_->data(), q_->stride(),
               size_.width(), size_.height(),
               params_->lambda, params_->eps_u, sigma, eta,
               *u_prev_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *it_tex_);

      // and primal update kernel
      k_primalUpdate
          <<<
            frag.dimGrid, frag.dimBlock
          >>> (u_->data(), u_prev_->data(), u_->stride(),
               size_.width(), size_.height(),
               params_->lambda, tau, lin_step,
               *u_tex_, *u0_tex_, *pu_tex_, *q_tex_, *ix_tex_, *xi_tex_);
    } // iters
    lin_step /= 1.2f;

  } // warps


  IMP_CUDA_CHECK();
}



} // namespace cu
} // namespace imp

