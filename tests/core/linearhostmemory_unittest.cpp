// system includes
#include <iostream>
#include <imp/core.h>
//#include <iu/iucore.h>
//#include <iu/iucutil.h>


int main(int argc, char** argv)
{
  unsigned int length = 1e7;

  // create linar hostbuffer
  imp::LinearHostMemory_8u_C1* h_8u_C1 = new imp::LinearHostMemory_8u_C1(length);
  imp::LinearHostMemory_32f_C1* h_32f_C1 = new imp::LinearHostMemory_32f_C1(length);

  // temporary variables to check the values
  imp::LinearHostMemory_8u_C1 check_8u_C1(h_8u_C1->length());
  imp::LinearHostMemory_32f_C1 check_32f_C1(h_32f_C1->length());

  // values to be set
  unsigned char val_8u = 1;
  float val_32f = 1.0f;

  // set host values
  h_8u_C1->setValue(val_8u);
  h_32f_C1->setValue(val_32f);

  // copy memory
  imp::copy(h_8u_C1, &check_8u_C1);
  imp::copy(h_32f_C1, &check_32f_C1);

  // check if the values are ok
  for (unsigned int i = 0; i<check_8u_C1.length(); ++i)
  {
    assert(*h_8u_C1->data(i) == val_8u);
    assert(*h_32f_C1->data(i) == val_32f);
    assert(*check_8u_C1.data(i) == val_8u);
    assert(*check_32f_C1.data(i) == val_32f);
  }

  //  cleanup
  delete(h_8u_C1);
  delete(h_32f_C1);

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;
}
