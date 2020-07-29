#include "packing_common.h"
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>

/* number of threads X number of cubes X data size */
float_type ***data_cubes;
int64_t *mmap_endpoints;

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

int mmap_counter = 0;
int mmap_rank = 0;

void *shared_alloc(size_t size)
{
    int protection = PROT_READ | PROT_WRITE;
    int visibility = MAP_SHARED;
    char fname[256];
    snprintf(fname, 255, "/dev/shm/ghex_%d_%.4d", mmap_rank, mmap_counter);
    mmap_counter++;
    
    int fd = open(fname, O_CREAT | O_RDWR, 0600);
    if (fd<0){
        fprintf(stderr, "open(%s) failed with error: %s\n", fname, strerror(errno));
        exit(1);
    }
    if(ftruncate(fd, size) < 0){
        fprintf(stderr, "ftruncate(%s) failed with error: %s\n", fname, strerror(errno));
        exit(1);
    }

    int *retval = (int*)mmap(NULL, size, protection, visibility, fd, 0);
    close(fd);
    return retval;
}

void *shared_attach(size_t size, int rank, int id)
{
    int protection = PROT_READ | PROT_WRITE;
    int visibility = MAP_SHARED;
    char fname[256];
    snprintf(fname, 255, "/dev/shm/ghex_%d_%.4d", rank, id);
    
    int fd = open(fname, O_RDWR, 0600);
    if (fd<0){
        fprintf(stderr, "open(%s) failed with error: %s\n", fname, strerror(errno));
        exit(1);
    }

    int *retval = (int*)mmap(NULL, size, protection, visibility, fd, 0);
    close(fd);
    return retval;
}

void shared_cleanup(int id)
{
    char fname[256];
    snprintf(fname, 255, "/dev/shm/ghex_%d_%.4d", mmap_rank, id);
    if(unlink(fname)<0){
        fprintf(stderr, "unlink(%s) failed with error: %s\n", fname, strerror(errno));
    }
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
    mmap_endpoints  = (int64_t*)malloc(sizeof(int64_t*)*num_ranks*num_fields);
    memset(mmap_endpoints, 0, sizeof(int64_t)*num_ranks*num_fields);

    /* init allocator */
    mmap_rank = rank;
    mmap_counter = 0;

    {
        int    coords[3];
        size_t memsize;

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
            data_cubes[rank][fi] = (float_type*)shared_alloc(memsize);
            memset(data_cubes[rank][fi], 0, memsize);

            /* share the memory */
            {
                mmap_endpoints[rank*num_fields + fi] = mmap_counter-1;
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

        /* exchange pointer information with node-local ranks */
        MPI_Allgather(MPI_IN_PLACE, num_fields, MPI_INT64_T, mmap_endpoints, num_fields, MPI_INT64_T, MPI_COMM_WORLD);

        /* mmap the peers */
        for(int ri=0; ri<num_ranks; ri++){

            if(rank==ri) continue;

            data_cubes[ri]  = (float_type**)malloc(sizeof(float_type**)*num_fields);
            for(int fi=0; fi<num_fields; fi++){
                data_cubes[ri][fi] = (float_type*)shared_attach(memsize, ri, fi);
            }
        }

        /* We can remove the files from /dev/shm immediately after the peers have mmaped them */
        /* The memory will remain mapped, and will be automatically freed after the program ends */
        MPI_Barrier(MPI_COMM_WORLD);
        for(int fi=0; fi<num_fields; fi++){
            shared_cleanup(fi);
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
