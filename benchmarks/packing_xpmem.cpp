#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

extern "C" {
#include <xpmem.h>
}

using float_type = float;

const int local_dims[3] = {128, 128, 128};
const int halo = 5;
const int num_fields = 8;
const int num_repetitions = 200;

int dims[3] = {0};
size_t dimx, dimy, dimz;

/* number of threads X number of cubes X data size */
float_type ***data_cubes;
xpmem_segid_t *xpmem_endpoints;

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

inline void rank2coord(int dims[3], int rank, int coord[3])
{
    int tmp = rank;    coord[0] = tmp%dims[0];
    tmp = tmp/dims[0]; coord[1] = tmp%dims[1];
    tmp = tmp/dims[1]; coord[2] = tmp;
}
  
inline int coord2rank(int dims[3], int coord0, int coord1, int coord2)
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

inline void __attribute__ ((always_inline)) x_copy_seq(const int thrid, const int coords[3],
    int *, int k, int nbz, int j, int nby, int id)
{
    int nb, i, dst_k, dst_j, dst_i;
    float *dst, *src;

    src = data_cubes[thrid][id];

    if(nbz==-1)      dst_k = k + local_dims[2];
    else if(nbz==1)  dst_k = k - local_dims[2];
    else             dst_k = k;
    
    if(nby==-1)      dst_j = j + local_dims[1];
    else if(nby==1)  dst_j = j - local_dims[1];
    else             dst_j = j;

    nb  = coord2rank(dims, coords[0]-1, coords[1]+nby, coords[2]+nbz);
    dst = data_cubes[nb][id];
    dst_i = local_dims[0];
#pragma ivdep
    for(i=halo; i<2*halo; i++){
        dst[dst_k*dimx*dimy + dst_j*dimx + dst_i + i] = src[k*dimx*dimy + j*dimx + i];
    }

    nb  = coord2rank(dims, coords[0]+0, coords[1]+nby, coords[2]+nbz);
    if(nb != thrid) {
        dst = data_cubes[nb][id];
#pragma ivdep
        for(i=halo; i<halo + local_dims[0]; i++){
            dst[dst_k*dimx*dimy + dst_j*dimx + i] = src[k*dimx*dimy + j*dimx + i];
        }
    }

    nb  = coord2rank(dims, coords[0]+1, coords[1]+nby, coords[2]+nbz);
    dst = data_cubes[nb][id];
    dst_i = -local_dims[0];
#pragma ivdep
    for(i=local_dims[0]; i<halo + local_dims[0]; i++){
        dst[dst_k*dimx*dimy + dst_j*dimx + dst_i + i] = src[k*dimx*dimy + j*dimx + i];
    }
}

inline void __attribute__ ((always_inline)) x_verify(const int thrid, const int coords[3], 
    int *errors, int k, int nbz, int j, int nby, int id)
{
    int nb, i, dst_k, dst_j, dst_i;
    float *dst;

    dst = data_cubes[thrid][id];

    if(nbz==-1)      dst_k = k - halo;
    else if(nbz==1)  dst_k = k + halo;
    else             dst_k = k;
    
    if(nby==-1)      dst_j = j - halo;
    else if(nby==1)  dst_j = j + halo;
    else             dst_j = j;

    nb  = coord2rank(dims, coords[0]-1, coords[1]+nby, coords[2]+nbz);
    dst_i = -halo;
    for(i=halo; i<2*halo; i++){
        *errors += dst[dst_k*dimx*dimy + dst_j*dimx + dst_i + i] != nb+1;
    }

    nb  = coord2rank(dims, coords[0]+0, coords[1]+nby, coords[2]+nbz);
    for(i=halo; i<halo + local_dims[0]; i++){
        *errors += dst[dst_k*dimx*dimy + dst_j*dimx + i] != nb+1;
    }

    nb  = coord2rank(dims, coords[0]+1, coords[1]+nby, coords[2]+nbz);
    dst_i = halo;
    for(i=local_dims[0]; i<halo + local_dims[0]; i++){
        *errors += dst[dst_k*dimx*dimy + dst_j*dimx + dst_i + i] != nb+1;
    }
}

#define Y_BLOCK(XBL, thrid, coords, arg, k, nbz, id)     \
    {                                                           \
        int j;                                                  \
        nby = -1;                                               \
        for(j=halo; j<2*halo; j++){                             \
            XBL(thrid, coords, arg, k, nbz, j, nby, id); \
        }                                                       \
        nby = 0;                                                \
        for(j=halo; j<halo + local_dims[1]; j++){               \
            XBL(thrid, coords, arg, k, nbz, j, nby, id); \
        }                                                       \
        nby = 1;                                                \
        for(j=local_dims[1]; j<halo + local_dims[1]; j++){      \
            XBL(thrid, coords, arg, k, nbz, j, nby, id); \
        }                                                       \
    }                                                           \


#define Z_BLOCK(XBL, thrid, coords, arg, id)                     \
    {                                                                   \
        int k;                                                          \
        nbz = -1;                                                       \
        for(k=halo; k<2*halo; k++){                                     \
            Y_BLOCK(XBL, thrid, coords, arg, k, nbz, id);        \
        }                                                               \
        nbz = 0;                                                        \
        for(k=halo; k<halo + local_dims[2]; k++){                       \
            Y_BLOCK(XBL, thrid, coords, arg, k, nbz, id);        \
        }                                                               \
        nbz = 1;                                                        \
        for(k=local_dims[2]; k<halo + local_dims[2]; k++){              \
            Y_BLOCK(XBL, thrid, coords, arg, k, nbz, id);        \
        }                                                               \
    }

#define PRINT_CUBE(data)                                                \
    {                                                                   \
        for(size_t k=0; k<dimz; k++){                                   \
            for(size_t j=0; j<dimy; j++){                               \
                for(size_t i=0; i<dimx; i++){                           \
                    printf("%f ", data[k*dimx*dimy + j*dimx + i]);      \
                }                                                       \
                printf("\n");                                           \
            }                                                           \
            printf("\n");                                               \
        }                                                               \
    }

int main(int argc, char *argv[])
{    
    int num_ranks = 1, rank;

    /* make a cartesian thread space */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Dims_create(num_ranks, 3, dims);
    if(0 == rank) printf("cart dims %d %d %d\n", dims[0], dims[1], dims[2]);

    /* allocate shared data structures */
    data_cubes  = (float_type***)malloc(sizeof(float_type***)*num_ranks);
    xpmem_endpoints  = (xpmem_segid_t*)malloc(sizeof(xpmem_segid_t*)*num_ranks*num_fields);
    memset(xpmem_endpoints, 0, sizeof(xpmem_segid_t*)*num_ranks*num_fields);

    {
        int    coords[3];
        void **ptr;
        size_t memsize;
        int nby, nbz;

        /* create a cartesian periodic rank world */
        rank2coord(dims, rank, coords);

        /* compute data size */
        dimx = (local_dims[0] + 2*halo);
        dimy = (local_dims[1] + 2*halo);
        dimz = (local_dims[2] + 2*halo);
        memsize = sizeof(float_type)*dimx*dimy*dimz;

        /* allocate per-thread fields */
        data_cubes[rank]  = (float_type**)malloc(sizeof(float_type**)*num_fields);
        for(int fi=0; fi<num_fields; fi++){

            /* allocate data and halos */
            ptr = (void**)(data_cubes[rank]+fi); 
            posix_memalign(ptr, 4096, memsize);
            memset(data_cubes[rank][fi], 0, memsize);

            /* share the memory */
            {
                int pagesize = getpagesize();
                int xpmemsize = memsize;
                if(memsize%pagesize) {
                    xpmemsize = pagesize*((memsize/pagesize)+1);
                }
            
                xpmem_endpoints[rank*num_fields + fi] = xpmem_make(data_cubes[rank][fi], xpmemsize, XPMEM_PERMIT_MODE, (void*)0666);
                if(-1 == xpmem_endpoints[rank*num_fields + fi]){
                    printf("failed to register xpmem segment\n");
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
            
            /* init domain data: owner thread id */
            for(size_t k=halo; k<dimz-halo; k++){
                for(size_t j=halo; j<dimy-halo; j++){
                    for(size_t i=halo; i<dimx-halo; i++){
                        data_cubes[rank][fi][k*dimx*dimy + j*dimx + i] = rank+1;
                    }
                }
            }
        }

        /* exchange xpmem endpoints with node-local ranks */
        MPI_Allgather(MPI_IN_PLACE, num_fields, MPI_INT64_T, xpmem_endpoints, num_fields, MPI_INT64_T, MPI_COMM_WORLD);
        
        /* xpmem segments to pointers */

        for(int ri=0; ri<num_ranks; ri++){

            if(rank==ri) continue;

            data_cubes[ri]  = (float_type**)malloc(sizeof(float_type**)*num_fields);
            for(int fi=0; fi<num_fields; fi++){

                int pagesize = getpagesize();
                int xpmemsize = memsize;
                if(memsize%pagesize) {
                    xpmemsize = pagesize*((memsize/pagesize)+1);
                }

                struct xpmem_addr addr;
                addr.apid = xpmem_get(xpmem_endpoints[ri*num_fields + fi], XPMEM_RDWR, XPMEM_PERMIT_MODE, (void*)0666);
                addr.offset = 0;

                data_cubes[ri][fi] = (float_type*)xpmem_attach(addr, xpmemsize, NULL);
            }
        }
    
        /* warmup */
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i=0; i<10; i++){
            for(int fi=0; fi<num_fields; fi++){
                Z_BLOCK(x_copy_seq, rank, coords, NULL, fi);
            }
        }

        /* halo exchange */
        MPI_Barrier(MPI_COMM_WORLD);
        tic();
        for(int i=0; i<num_repetitions; i++){
            for(int fi=0; fi<num_fields; fi++){
                Z_BLOCK(x_copy_seq, rank, coords, NULL, fi);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        bytes = 2*num_repetitions*num_ranks*num_fields*(dimx*dimy*dimz-local_dims[0]*local_dims[1]*local_dims[2])*sizeof(float_type);
        if(rank==0) toc();

        /* verify that the data is correct */
        {
            int errors = 0;
            for(int fi=0; fi<num_fields; fi++){
                Z_BLOCK(x_verify, rank, coords, &errors, 0);
            }
            if(errors) printf("ERROR: %d values do not validate\n", errors);
        }
    }
        
    MPI_Finalize();
}
