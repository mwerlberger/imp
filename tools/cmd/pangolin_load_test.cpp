#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <pangolin/image.h>
//#include <pangolin/image_load.h>
//#define BUILD_PANGOLIN_GUI
//#include <pangolin/config.h>
#include <pangolin/pangolin.h>

#include <imp/core/roi.hpp>
#include <imp/core/image_raw.hpp>
#include <imp/core/image_cv.hpp>
#include <imp/cucore/cu_image_gpu.cuh>
#include <imp/bridge/pangolin/imread.hpp>



void setImageData(unsigned char * imageArray, int size){
  for(int i = 0 ; i < size;i++) {
    imageArray[i] = (unsigned char)(rand()/(RAND_MAX/255.0));
  }
}

int main( int /*argc*/, char* argv[] )
{
  google::InitGoogleLogging(argv[0]);
  const std::string filename("/home/mwerlberger/data/std/Lena.png");

  // try to load an image with pangolin first
//  pangolin::TypedImage im = pangolin::LoadImage(filename,
//                                                pangolin::ImageFileType::ImageFileTypePng);

  std::shared_ptr<imp::ImageRaw8uC1> im_8uC1;
  imp::pangolinBridgeLoad(im_8uC1, filename, imp::PixelOrder::gray);

  VLOG(2) << "Read Lena (png) from " << filename
          << ": " << im_8uC1->width() << "x" << im_8uC1->height() << "(" << im_8uC1->pitch() << ")";
          //<< "; format: " << im.fmt.format;



  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Lena", im_8uC1->width(), im_8uC1->height());

  if (glewInit() != GLEW_OK )
  {
    LOG(ERROR) << "Unable to initialize GLEW." << std::endl;
  }

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  pangolin::View& container = pangolin::CreateDisplay()
      .SetBounds(0, 1.0f, 0, 1.0f, (double)im_8uC1->width()/im_8uC1->height());

  container.SetLayout(pangolin::LayoutEqual);
  pangolin::View& v = pangolin::CreateDisplay();
  v.SetAspect((double)im_8uC1->width()/im_8uC1->height());
  container.AddDisplay(v);

  // texture to dispaly
  pangolin::GlTexture tex8(im_8uC1->width(), im_8uC1->height(), GL_LUMINANCE8);


//  pangolin::OpenGlRenderState s_cam(
//        pangolin::ProjectionMatrix(img->width(), img->height(), 500, 500, img->width()/2, img->height()/2, 0.1, 1000),
//        pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
//        );

//  // Aspect ratio allows us to constrain width and height whilst fitting within specified
//  // bounds. A positive aspect ratio makes a view 'shrink to fit' (introducing empty bars),
//  // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
//  pangolin::View& d_cam = pangolin::Display("cam")
//      .SetBounds(0,1.0f,0,1.0f,-(float)img->width()/(float)img->height())
//      .SetHandler(new pangolin::Handler3D(s_cam));

//  // This view will take up no more than a third of the windows width or height, and it
//  // will have a fixed aspect ratio to match the image that it will display. When fitting
//  // within the specified bounds, push to the top-left (as specified by SetLock).
//  pangolin::View& d_image = pangolin::Display("image")
//      .SetBounds(2/3.0f,1.0f,0,1/3.0f,(float)img->width()/(float)img->height())
//      .SetLock(pangolin::LockLeft, pangolin::LockTop);

//  //  std::cout << "Resize the window to experiment with SetBounds, SetLock and SetAspect." << std::endl;
//  //  std::cout << "Notice that the teapots aspect is maintained even though it covers the whole screen." << std::endl;

//  //  const int width =  64;
//  //  const int height = 48;

//  //  unsigned char* imageArray = new unsigned char[3*width*height];
//  pangolin::GlTexture imageTexture(img->width(), img->height(),GL_LUMINANCE8,false,0,GL_LUMINANCE8,GL_UNSIGNED_BYTE);

  // Default hooks for exiting (Esc) and fullscreen (tab).
  while(!pangolin::ShouldQuit())
  {
    // drawing
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(1,1,1);

    container.Activate();
    tex8.Upload(im_8uC1->data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);
    tex8.RenderToViewportFlipY();

//    d_cam.Activate(s_cam);
//    glColor3f(1.0,1.0,1.0);
//    pangolin::glDrawColouredCube();

    //Set some random image data and upload to GPU
//    setImageData(imageArray,3*width*height);
//    imageTexture.Upload(im.ptr, GL_LUMINANCE, GL_UNSIGNED_BYTE);

    //display the image
//    d_image.Activate();
    //glColor3f(1.0,1.0,1.0);
//    imageTexture.RenderToViewport();

    pangolin::FinishFrame();
  }

//  delete[] imageArray;

  //  return 0;
}


//int main(int /*argc*/, char** /*argv*/)
//{
//  try
//  {
//    std::shared_ptr<imp::cu::ImageGpu32fC1> im;
//    imp::cu::cvBridgeLoad(im, "/home/mwerlberger/data/std/cones/im2.ppm",
//                          imp::PixelOrder::gray);

//    std::unique_ptr<imp::cu::ImageGpu32fC1> edges(
//          new imp::cu::ImageGpu32fC1(*im));

//    imp::cu::naturalEdges(*edges, *im, 1.f, 10.f, 0.7f);

//    imp::cu::cvBridgeShow("image", *im);
//    imp::cu::cvBridgeShow("edges", *edges, true);

//    cv::waitKey();
//  }
//  catch (std::exception& e)
//  {
//    std::cout << "[exception] " << e.what() << std::endl;
//    assert(false);
//  }

//  return EXIT_SUCCESS;

//}
