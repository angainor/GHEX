#include "packing_common.h"

/* number of threads X number of cubes X data size */
float_type ***data_cubes;

inline void __attribute__ ((always_inline)) x_copy_seq(const int rank, const int coords[3],
    int *, int k, int nbz, int j, int nby, int id)
{
    int nb, i, dst_k, dst_j, dst_i;
    float_type *dst, *src;

    src = data_cubes[rank][id];

    if(nbz==-1)      dst_k = k + local_dims[2];
    else if(nbz==1)  dst_k = k - local_dims[2];
    else             dst_k = k;
    
    if(nby==-1)      dst_j = j + local_dims[1];
    else if(nby==1)  dst_j = j - local_dims[1];
    else             dst_j = j;

    nb  = coord2rank(dims, coords[0]-1, coords[1]+nby, coords[2]+nbz);
    dst = data_cubes[nb][id];
    dst_i = local_dims[0];

#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(i=halo; i<2*halo; i++){
        dst[dst_k*dimx*dimy + dst_j*dimx + dst_i + i] = src[k*dimx*dimy + j*dimx + i];
    }

    if(nby!=0 || nbz!=0) {
        nb  = coord2rank(dims, coords[0]+0, coords[1]+nby, coords[2]+nbz);
        dst = data_cubes[nb][id];

#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for(i=halo; i<halo + local_dims[0]; i++){
            dst[dst_k*dimx*dimy + dst_j*dimx + i] = src[k*dimx*dimy + j*dimx + i];
        }
    }

    nb  = coord2rank(dims, coords[0]+1, coords[1]+nby, coords[2]+nbz);
    dst = data_cubes[nb][id];
    dst_i = -local_dims[0];

#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(i=local_dims[0]; i<halo + local_dims[0]; i++){
        dst[dst_k*dimx*dimy + dst_j*dimx + dst_i + i] = src[k*dimx*dimy + j*dimx + i];
    }
}

inline void __attribute__ ((always_inline)) x_verify(const int rank, const int coords[3], 
    int *errors, int k, int nbz, int j, int nby, int id)
{
    int nb, i, dst_k, dst_j, dst_i;
    float_type *dst;

    dst = data_cubes[rank][id];

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

int main(int argc, char* argv[])
{    
    int num_ranks = 1;

    if(argc!=4){
      fprintf(stderr, "Usage: <bench name> thread_space_dimensions\n");
      exit(1);
    }

    for(int i=0; i<3; i++){
      dims[i] = atoi(argv[i+1]);
      num_ranks *= dims[i];
    }
    omp_set_num_threads(num_ranks);

    /* make a cartesian thread space */
    printf("cart dims %d %d %d\n", dims[0], dims[1], dims[2]);

    /* allocate shared data structures */
    data_cubes  = (float_type***)malloc(sizeof(float_type***)*num_ranks);

    /* allocate per-thread data */
#pragma omp parallel
    {
        int    rank;
        int    coords[3];
        void **ptr;
        size_t memsize;

        /* create a cartesian periodic rank world */
        rank = omp_get_thread_num();
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
            posix_memalign(ptr, 64, memsize);
            memset(data_cubes[rank][fi], 0, memsize);
            
            /* init domain data: owner thread id */
            for(size_t k=halo; k<dimz-halo; k++){
                for(size_t j=halo; j<dimy-halo; j++){
                    for(size_t i=halo; i<dimx-halo; i++){
                        data_cubes[rank][fi][k*dimx*dimy + j*dimx + i] = rank+1;
                    }
                }
            }
        }

#pragma omp barrier
        /* warmup */
        for(int i=0; i<2; i++){
            for(int fi=0; fi<num_fields; fi++){
                Z_BLOCK(x_copy_seq, rank, coords, NULL, fi);
            }
        }

#pragma omp barrier
        tic();

        /* halo exchange */
        for(int i=0; i<num_repetitions; i++){
#pragma omp barrier
            for(int fi=0; fi<num_fields; fi++){
                Z_BLOCK(x_copy_seq, rank, coords, NULL, fi);
            }
        }

#pragma omp barrier
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
}