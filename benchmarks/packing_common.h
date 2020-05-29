#ifndef _INCLUDED_PACKING_COMMON_H_
#define _INCLUDED_PACKING_COMMON_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>

using float_type = float;
using size_type = int_fast64_t;

#define USE_MEMCPY
#define DIMX 128
const int local_dims[3] = {DIMX, DIMX, DIMX};
const int halo = 5;
const int num_fields = 8;
const int num_repetitions = 100;

static struct timeval tb, te;
double bytes = 0;
void tic(void)
{
    gettimeofday(&tb, NULL);
    bytes=0;
    fflush(stdout);
}

void toc(void)
{
    long s,u;
    double tt;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    tt=((double)s)*1000000+u;
    printf("time:                  %li.%.6lis\n", (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    printf("MB/s:                  %.3lf\n", bytes/tt);
    fflush(stdout);
}

inline void rank2coord(const int dims[3], int rank, int coord[3])
{
    int tmp = rank;    coord[0] = tmp%dims[0];
    tmp = tmp/dims[0]; coord[1] = tmp%dims[1];
    tmp = tmp/dims[1]; coord[2] = tmp;
}
  
inline int coord2rank(const int dims[3], int coord0, int coord1, int coord2)
{
    // periodicity
    if(coord0<0) coord0 = dims[0]-1;
    if(coord1<0) coord1 = dims[1]-1;
    if(coord2<0) coord2 = dims[2]-1;
    if(coord0==dims[0])  coord0 = 0;
    if(coord1==dims[1])  coord1 = 0;
    if(coord2==dims[2])  coord2 = 0;
    return (coord2*dims[1] + coord1)*dims[0] + coord0;
}

inline int id2nbid(const int i, const int j, const int k)
{
    return ((k+1)*3+j+1)*3+i+1;
}


#define FILL_IT_SPACE()                         \
    it_space[nbid][0] = zlo;                    \
    it_space[nbid][1] = zhi;                    \
    it_space[nbid][2] = ylo;                    \
    it_space[nbid][3] = yhi;                    \
    it_space[nbid][4] = xlo;                    \
    it_space[nbid][5] = xhi;                    \

inline void X_RANGE_SRC(size_type **it_space, size_type zlo, size_type zhi, size_type nbz, size_type ylo, size_type yhi, size_type nby)
{
    size_type xlo, xhi, nbx, nbid;
    xlo = halo;
    xhi = 2*halo;
    nbx = 1;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();

    xlo = halo;
    xhi = halo + local_dims[0];
    nbx = 0;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();

    xlo = local_dims[0];
    xhi = halo + local_dims[0];
    nbx = -1;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();
}

#define Y_RANGE_SRC(it_space, zlo, zhi, nbz)				\
    {                                                                   \
        X_RANGE_SRC(it_space, zlo, zhi, nbz, halo, 2*halo, 1);		\
        X_RANGE_SRC(it_space, zlo, zhi, nbz, halo, halo + local_dims[1], 0); \
        X_RANGE_SRC(it_space, zlo, zhi, nbz, local_dims[1], halo + local_dims[1], -1); \
    }

#define Z_RANGE_SRC(it_space)						\
    {                                                                   \
        Y_RANGE_SRC(it_space, halo, 2*halo, 1);                         \
        Y_RANGE_SRC(it_space, halo, halo + local_dims[2], 0);		\
        Y_RANGE_SRC(it_space, local_dims[2], halo + local_dims[2], -1);	\
    }

inline void X_RANGE_DST(size_type **it_space, size_type zlo, size_type zhi, size_type nbz, size_type ylo, size_type yhi, size_type nby)
{
    size_type xlo, xhi, nbx, nbid;
    xlo = 0;
    xhi = halo;
    nbx = -1;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();

    xlo = halo;
    xhi = halo + local_dims[0];
    nbx = 0;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();

    xlo = halo + local_dims[0];
    xhi = 2*halo + local_dims[0];
    nbx = 1;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();
}

#define Y_RANGE_DST(it_space, zlo, zhi, nbz)				\
    {                                                                   \
        X_RANGE_DST(it_space, zlo, zhi, nbz, 0, halo, -1);		\
        X_RANGE_DST(it_space, zlo, zhi, nbz, halo, halo + local_dims[1], 0); \
        X_RANGE_DST(it_space, zlo, zhi, nbz, halo+local_dims[1], 2*halo + local_dims[1], 1); \
    }

#define Z_RANGE_DST(it_space)						\
    {                                                                   \
        Y_RANGE_DST(it_space, 0, halo, -1);				\
        Y_RANGE_DST(it_space, halo, halo + local_dims[2], 0);		\
        Y_RANGE_DST(it_space, halo + local_dims[2], 2*halo + local_dims[2], 1); \
    }

#define PRINT_CUBE(data)                                                \
    {                                                                   \
        for(size_type k=0; k<dimz; k++){                                \
            for(size_type j=0; j<dimy; j++){                            \
                for(size_type i=0; i<dimx; i++){                        \
                    printf("%f ", data[k*dimx*dimy + j*dimx + i]);      \
                }                                                       \
                printf("\n");                                           \
            }                                                           \
            printf("\n");                                               \
        }                                                               \
    }

#endif /*  _INCLUDED_PACKING_COMMON_H_ */
