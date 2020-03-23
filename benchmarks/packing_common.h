#ifndef _INCLUDED_PACKING_COMMON_H_
#define _INCLUDED_PACKING_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

using float_type = float;

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

#endif /*  _INCLUDED_PACKING_COMMON_H_ */
