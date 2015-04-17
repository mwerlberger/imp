#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>

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
#include <imp/bridge/opencv/cu_cv_bridge.hpp>



void setImageData(unsigned char * imageArray, int size){
  for(int i = 0 ; i < size;i++) {
    imageArray[i] = (unsigned char)(rand()/(RAND_MAX/255.0));
  }
}

int main( int /*argc*/, char* argv[] )
{
  pangolin::CreateWindowAndBind("Main",640,480);
  glEnable(GL_DEPTH_TEST);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
      pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
  );

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
          .SetHandler(&handler);

  while( !pangolin::ShouldQuit() )
  {
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      d_cam.Activate(s_cam);

      // Render OpenGL Teapot
      pangolin::glDrawColouredCube();

      // Swap frames and Process Events
      pangolin::FinishFrame();
  }

  return 0;

//  // Create OpenGL window in single line
//  pangolin::CreateWindowAndBind("Main",640,480);

//  // 3D Mouse handler requires depth testing to be enabled
//  glEnable(GL_DEPTH_TEST);

//  pangolin::OpenGlRenderState s_cam(
//    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
//    pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
//  );

//  // Aspect ratio allows us to constrain width and height whilst fitting within specified
//  // bounds. A positive aspect ratio makes a view 'shrink to fit' (introducing empty bars),
//  // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
//  pangolin::View& d_cam = pangolin::Display("cam")
//      .SetBounds(0,1.0f,0,1.0f,-640/480.0)
//      .SetHandler(new pangolin::Handler3D(s_cam));

//  // This view will take up no more than a third of the windows width or height, and it
//  // will have a fixed aspect ratio to match the image that it will display. When fitting
//  // within the specified bounds, push to the top-left (as specified by SetLock).
//  pangolin::View& d_image = pangolin::Display("image")
//      .SetBounds(2/3.0f,1.0f,0,1/3.0f,640.0/480)
//      .SetLock(pangolin::LockLeft, pangolin::LockTop);

//  std::cout << "Resize the window to experiment with SetBounds, SetLock and SetAspect." << std::endl;
//  std::cout << "Notice that the teapots aspect is maintained even though it covers the whole screen." << std::endl;

//  const int width =  64;
//  const int height = 48;

//  unsigned char* imageArray = new unsigned char[3*width*height];
//  pangolin::GlTexture imageTexture(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

//  // Default hooks for exiting (Esc) and fullscreen (tab).
//  while(!pangolin::ShouldQuit())
//  {
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    d_cam.Activate(s_cam);

//    glColor3f(1.0,1.0,1.0);
//    pangolin::glDrawColouredCube();

//    //Set some random image data and upload to GPU
//    setImageData(imageArray,3*width*height);
//    imageTexture.Upload(imageArray,GL_RGB,GL_UNSIGNED_BYTE);

//    //display the image
//    d_image.Activate();
//    glColor3f(1.0,1.0,1.0);
//    imageTexture.RenderToViewport();

//    pangolin::FinishFrame();
//  }

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
