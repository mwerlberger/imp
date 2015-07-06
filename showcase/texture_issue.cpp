#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_imgproc/iterative_kernel_calls.cuh>

int main(int argc, char** argv)
{
  try
  {
    if (argc < 2)
    {
      std::cout << "usage: texture_issue input_image_filename [break_things_flag]";
      return EXIT_FAILURE;
    }

    std::string in_filename(argv[1]);

    cv::Mat image_8u = cv::imread(in_filename, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat image(image_8u.rows, image_8u.cols, CV_32FC1);
    image_8u.convertTo(image, CV_32F, 1./255.);
    cv::Mat result_image(image.rows, image.cols, CV_32FC1);

    cudaError cu_err;

    {
      imp::cu::ImageGpu32fC1::Ptr cu_image = std::make_shared<imp::cu::ImageGpu32fC1>(image.cols, image.rows);
      imp::cu::ImageGpu32fC1::Ptr cu_result_image = std::make_shared<imp::cu::ImageGpu32fC1>(image.cols, image.rows);
      IMP_CUDA_CHECK();
      std::cout << "cu_image: " << cu_image->pitch() << std::endl;
      std::cout << "image: " << image.step << std::endl;

      // copy image data to device
      cu_err = cudaMemcpy2D(cu_image->data(), cu_image->pitch(),
                            (void*)image.data, image.step,
                            image.cols*sizeof(float), image.rows,
                            cudaMemcpyHostToDevice);
      if (cu_err != cudaSuccess)
      {
        throw imp::cu::Exception("copy host -> device failed", cu_err, __FILE__, __FUNCTION__, __LINE__);
        return EXIT_FAILURE;
      }
      IMP_CUDA_CHECK();

      //
      //
      imp::cu::IterativeKernelCalls ikc;
      bool break_things = (argc>2) ? true : false;
      ikc.denoise(cu_result_image, cu_image, break_things);
      //
      //


      // copy image data to device
      cu_err = cudaMemcpy2D((void*)result_image.data, result_image.step,
                            cu_result_image->data(), cu_result_image->pitch(),
                            cu_result_image->rowBytes(), cu_result_image->height(),
                            cudaMemcpyDeviceToHost);
      if (cu_err != cudaSuccess)
      {
        throw imp::cu::Exception("copy device -> host failed", cu_err, __FILE__, __FUNCTION__, __LINE__);
        return EXIT_FAILURE;
      }
      IMP_CUDA_CHECK();
    }

    cv::imshow("input image", image);
    cv::imshow("roundtrip image", result_image);
    cv::waitKey();
  }
  catch (std::exception& e)
  {
    std::cout << "[exception] " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}
