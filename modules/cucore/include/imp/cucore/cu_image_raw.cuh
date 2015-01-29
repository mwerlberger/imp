#ifndef IMP_CUIMAGERAW_CUH
#define CUIMAGERAW_CUH

template<typename Pixel, imp::PixelType pixel_type>
class CuImageRaw : public Image
{
public:
  CuImageRaw();
  ~CuImageRaw();
};

#endif // CUIMAGERAW_CUH
