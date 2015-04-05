#ifndef IMP_CU_PINHOLE_CAMERA_CUH
#define IMP_CU_PINHOLE_CAMERA_CUH

#include <cuda_runtime_api.h>
#include <imp/core/pixel.hpp>
#include <imp/cucore/cu_matrix.cuh>

namespace imp {
namespace cu {

/**
 * @brief The PinholeCamera class implements a very simple pinhole camera model.
 *        Points in the image plane are denoted by the coordinate vectors (u,v) whereas
 *        world coordinates use the (x,y,z) nominclatur.
 *
 * @todo (MWE) marry off with other camera implementations (maybe with aslam_cv2 camera models? -> resolve Eigen vs. CUDA issues first)
 *
 */
class PinholeCamera
{
public:
  __host__ PinholeCamera() = default;
  __host__ ~PinholeCamera() = default;

  __host__ PinholeCamera(float fu, float fv, float cu, float cv)
    : f_(fu, fv)
    , c_(cu, cv)
  {
  }

  __host__ __device__ __forceinline__
  imp::cu::Matrix3f intrinsics()
  {
    imp::cu::Matrix3f K;
    K(0,0) = f_.x;
    K(0,1) = 0.0f;
    K(0,2) = c_.x;
    K(1,0) = 0.0f;
    K(1,1) = f_.y;
    K(1,2) = c_.y;
    K(2,0) = 0.0f;
    K(2,1) = 0.0f;
    K(2,2) = 1.0f;
    return K;
  }

  __host__ __device__ __forceinline__
  Vec32fC3 cam2world(const Vec32fC2& uv) const
  {
    return Vec32fC3((uv.x-c_.x)/f_.x,
                    (uv.y-c_.y)/f_.y,
                    1.0f);
  }

  __host__ __device__ __forceinline__
  Vec32fC2 world2cam(const Vec32fC3& p) const
  {
    return Vec32fC2(f_.x*p.x/p.z + c_.x,
                    f_.y*p.y/p.z + c_.y);
  }

  inline const Vec32fC2& f() {return f_;}
  inline const Vec32fC2& c() {return c_;}

  inline float fx() {return f_.x;}
  inline float fy() {return f_.y;}
  inline float cx() {return c_.x;}
  inline float cy() {return c_.y;}

private:
  Vec32fC2 f_; //!< focal length {fx, fy}
  Vec32fC2 c_; //!< principal point {cx, cy}

};

}
}

#endif // IMP_CU_PINHOLE_CAMERA_CUH

