#include "packing_common.h"
#include <iostream>
#include <omp.h>

const int dims[3] = {4,4,2};

// thread-private data accessed from pack/unpack routines
float_type **data_cubes;
int    buffer_pos[27] = {0};
size_t dimx, dimy, dimz;
int nby, nbz;
#pragma omp threadprivate(data_cubes)
#pragma omp threadprivate(buffer_pos)
#pragma omp threadprivate(dimx, dimy, dimz)
#pragma omp threadprivate(nby, nbz)

float_type ***compact_buffers;     // a large buffer to store all fields, for each neighbor, including self
float_type ****sequential_buffers; // num_fields small buffers for each neighbor, including self

inline void __attribute__ ((always_inline)) x_pack_seq(int *it_space, float_type __restrict__ *src, float_type __restrict__ *dst)
{
    int buffer_pos = 0;

    for(int k=it_space[0]; k<it_space[1]; k++){
        for(int j=it_space[2]; j<it_space[3]; j++){
            if(1){
                int len = it_space[5]-it_space[4];
                memcpy(dst+buffer_pos, src+k*dimx*dimy + j*dimx + it_space[4], sizeof(float_type)*len);
                buffer_pos+=len;
            } else {
                for(int i=it_space[4]; i<it_space[5]; i++){
                    dst[buffer_pos++] = src[k*dimx*dimy + j*dimx + i];
                }
            }
        }
    }
}

inline void __attribute__ ((always_inline)) x_unpack_seq(int *it_space, float_type __restrict__ *src, float_type __restrict__ *dst)
{
    int buffer_pos = 0;

    for(int k=it_space[0]; k<it_space[1]; k++){
        for(int j=it_space[2]; j<it_space[3]; j++){
            if(0){
                int len = it_space[5]-it_space[4];
                memcpy(dst+k*dimx*dimy + j*dimx + it_space[4], src+buffer_pos, sizeof(float_type)*len);
                buffer_pos+=len;
            } else {
                for(int i=it_space[4]; i<it_space[5]; i++){
                    dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos++];
                }
            }
        }
    }
}

// inline void __attribute__ ((always_inline)) x_pack_compact(const int rank, const int [3],
//     int *, int k, int nbz, int j, int nby, int id)
// {
// }


#define FILL_IT_SPACE()                         \
    it_space[nbid][0] = zlo;                    \
    it_space[nbid][1] = zhi;                    \
    it_space[nbid][2] = ylo;                    \
    it_space[nbid][3] = yhi;                    \
    it_space[nbid][4] = xlo;                    \
    it_space[nbid][5] = xhi;                    \

inline void X_RANGE(int **it_space, int zlo, int zhi, int nbz, int ylo, int yhi, int nby)
{
    int xlo, xhi, nbx, nbid;
    xlo = halo;
    xhi = 2*halo;
    nbx = -1;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();

    xlo = halo;
    xhi = halo + local_dims[0];
    nbx = 0;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();

    xlo = local_dims[0];
    xhi = halo + local_dims[0];
    nbx = 1;
    nbid = id2nbid(nbx, nby, nbz);
    FILL_IT_SPACE();
}

#define Y_RANGE(it_space, zlo, zhi, nbz)                                \
    {                                                                   \
        X_RANGE(it_space, zlo, zhi, nbz, halo, 2*halo, -1);             \
        X_RANGE(it_space, zlo, zhi, nbz, halo, halo + local_dims[1], 0); \
        X_RANGE(it_space, zlo, zhi, nbz, local_dims[1], halo + local_dims[1], 1); \
    }

#define Z_RANGE(it_space)                                               \
    {                                                                   \
        Y_RANGE(it_space, halo, 2*halo, -1);                            \
        Y_RANGE(it_space, halo, halo + local_dims[2], 0);               \
        Y_RANGE(it_space, local_dims[2], halo + local_dims[2], 1);      \
    }


inline void X_RANGE_UNPACK(int **it_space, int zlo, int zhi, int nbz, int ylo, int yhi, int nby)
{
    int xlo, xhi, nbx, nbid;
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

#define Y_RANGE_UNPACK(it_space, zlo, zhi, nbz)                         \
    {                                                                   \
        X_RANGE_UNPACK(it_space, zlo, zhi, nbz, 0, halo, -1);           \
        X_RANGE_UNPACK(it_space, zlo, zhi, nbz, halo, halo + local_dims[1], 0); \
        X_RANGE_UNPACK(it_space, zlo, zhi, nbz, halo+local_dims[1], 2*halo + local_dims[1], 1); \
    }

#define Z_RANGE_UNPACK(it_space)                                        \
    {                                                                   \
        Y_RANGE_UNPACK(it_space, 0, halo, -1);                          \
        Y_RANGE_UNPACK(it_space, halo, halo + local_dims[2], 0);        \
        Y_RANGE_UNPACK(it_space, halo + local_dims[2], 2*halo + local_dims[2], 1); \
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

//TEST(packing, strategies) {
int main(){
    int num_ranks = 1;

#pragma omp parallel
    {
#pragma omp master
        num_ranks = omp_get_num_threads();
    }

    // thread-shared for communication
    compact_buffers = (float_type***)malloc(sizeof(float_type***)*num_ranks);
    sequential_buffers = (float_type****)malloc(sizeof(float_type****)*num_ranks);

#pragma omp parallel
    {
        int    rank;
        int    coords[3];
        void **ptr;
        size_t memsize, halosize;

        dimx = (local_dims[0] + 2*halo);
        dimy = (local_dims[1] + 2*halo);
        dimz = (local_dims[2] + 2*halo);
        memsize = sizeof(float_type)*dimx*dimy*dimz;

        // create a cartesian periodic rank world
        rank = omp_get_thread_num();
        rank2coord(dims, rank, coords);

        int **iteration_spaces_pack;              // for each thread, for each nbor, 6 ints
        int **iteration_spaces_unpack;              // for each thread, for each nbor, 6 ints

        // explicitly create halo iteration spaces
        iteration_spaces_pack = (int**)malloc(sizeof(int**)*27);
        iteration_spaces_unpack = (int**)malloc(sizeof(int**)*27);
        for(int nbid=0; nbid<27; nbid++){
            iteration_spaces_pack[nbid] = (int*)malloc(sizeof(int*)*6);
            iteration_spaces_unpack[nbid] = (int*)malloc(sizeof(int*)*6);
        }
        
        Z_RANGE(iteration_spaces_pack);        
        Z_RANGE_UNPACK(iteration_spaces_unpack);        

        // compute memory footprint
        double n_halo_values = 0;
        for(int nbid=0; nbid<27; nbid++){
            int *is = iteration_spaces_unpack[nbid];
            if(nbid==13) continue;
            n_halo_values += (is[1]-is[0])*(is[3]-is[2])*(is[5]-is[4]);
        }

        // compute our nbid in neighbor arrays
        int nb[27] = {};
        int local_nbid = 0;
        for(int nbz=-1; nbz<=1; nbz++){
            for(int nby=-1; nby<=1; nby++){
                for(int nbx=-1; nbx<=1; nbx++){
                    nb[local_nbid]   = coord2rank(dims, coords[0]-nbx, coords[1]-nby, coords[2]-nbz);
                    local_nbid++;
                }
            }
        }

#pragma omp barrier

        // allocate per-thread fields
        data_cubes  = (float_type**)malloc(sizeof(float_type**)*num_fields);
        for(int fi=0; fi<num_fields; fi++){

            // allocate data and halos
            ptr = (void**)(data_cubes+fi);
            posix_memalign(ptr, 4096, memsize);
            memset(data_cubes[fi], 0, memsize);

            // init domain data: owner thread id
            for(size_t k=halo; k<dimz-halo; k++){
                for(size_t j=halo; j<dimy-halo; j++){
                    for(size_t i=halo; i<dimx-halo; i++){
                        data_cubes[fi][k*dimx*dimy + j*dimx + i] = rank+1;
                    }
                }
            }
        }

        // compute max halo buffer size
        halosize =
            std::max(local_dims[0]*local_dims[1],
                std::max(local_dims[0]*local_dims[2], local_dims[1]*local_dims[2]))*halo*sizeof(float_type);

        // allocate compact pack buffers
        compact_buffers[rank] = (float_type**)calloc(sizeof(float_type**), 27);
        for(int i=0; i<27; i++){
            ptr = (void**)(compact_buffers[rank]+i);
            posix_memalign(ptr, 4096, halosize*num_fields);
            memset(compact_buffers[rank][i], 0, halosize*num_fields);
        }

        // allocate individual pack buffers
        sequential_buffers[rank] = (float_type***)calloc(sizeof(float_type***), 27);
        for(int j=0; j<27; j++){
            sequential_buffers[rank][j] = (float_type**)calloc(sizeof(float_type**), num_fields);
            for(int i=0; i<num_fields; i++){
                ptr = (void**)(sequential_buffers[rank][j]+i);
                posix_memalign(ptr, 4096, halosize);
                memset(sequential_buffers[rank][j][i], 0, halosize);
            }
        }

        // packing loop: put all data into split comm buffers - separate for each field
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    if(nbid==13) continue;
                    x_pack_seq(iteration_spaces_pack[nbid], data_cubes[fid], sequential_buffers[rank][nbid][fid]);
                }
            }
        }
#pragma omp barrier
        bytes = n_halo_values*num_fields*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("packing (sequence)\n");
            toc();
        }


        // packing loop: put all data into split comm buffers - separate for each field
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    if(nbid==13) continue;
                    x_unpack_seq(iteration_spaces_unpack[nbid], sequential_buffers[nb[nbid]][26-nbid][fid], data_cubes[fid]);
                }
            }
        }
#pragma omp barrier
        bytes = n_halo_values*num_fields*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("unpacking (sequence)\n");
            toc();
        }


        /* verify that the data is correct */
        {
            int errors = 0;
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    int *it_space = iteration_spaces_unpack[nbid];
                    for(int k=it_space[0]; k<it_space[1]; k++){
                        for(int j=it_space[2]; j<it_space[3]; j++){
                            for(int i=it_space[4]; i<it_space[5]; i++){
                                if(data_cubes[fid][k*dimx*dimy + j*dimx + i] != nb[nbid]+1) errors++;
                            }
                        }
                    }
                }
            }
            if(errors) printf("ERROR: %d values do not validate\n", errors);
        }

        // combined
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    if(nbid==13) continue;
                    x_pack_seq(iteration_spaces_pack[nbid], data_cubes[fid], sequential_buffers[rank][nbid][fid]);
                    x_unpack_seq(iteration_spaces_unpack[nbid], sequential_buffers[nb[nbid]][26-nbid][fid], data_cubes[fid]);
                }
            }
        }
#pragma omp barrier
        bytes = n_halo_values*num_fields*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("pack+unpack (sequence)\n");
            toc();
        }
    }
}
