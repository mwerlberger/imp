#ifndef __KERNEL_REDUCE_H__
#define __KERNEL_REDUCE_H__

template <class T>
void reduce(int size, int threads, int blocks,
            T *d_idata, T *d_odata);

#endif
