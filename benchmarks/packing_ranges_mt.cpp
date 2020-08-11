#include "packing_common.h"
#include <iostream>

// thread-private data accessed from pack/unpack routines
float_type **data_cubes;
int    buffer_pos[27] = {0};
int nby, nbz;
#pragma omp threadprivate(data_cubes)
#pragma omp threadprivate(buffer_pos)
#pragma omp threadprivate(dimx, dimy, dimz)
#pragma omp threadprivate(nby, nbz)

float_type ***compact_buffers;     // a large buffer to store all fields, for each neighbor, including self
float_type ****sequential_buffers; // num_fields small buffers for each neighbor, including self

inline void __attribute__ ((always_inline)) pack_seq(size_type *it_space, float_type __restrict__ *src, float_type __restrict__ *dst)
{
    size_type buffer_pos = 0;

    for(size_type k=it_space[0]; k<it_space[1]; k++){
        for(size_type j=it_space[2]; j<it_space[3]; j++){
            if(1){
                size_type len = it_space[5]-it_space[4];
                memcpy(dst+buffer_pos, src+k*dimx*dimy + j*dimx + it_space[4], sizeof(float_type)*len);
                buffer_pos+=len;
            } else {
                for(size_type i=it_space[4]; i<it_space[5]; i++){
                    dst[buffer_pos++] = src[k*dimx*dimy + j*dimx + i];
                }
            }
        }
    }
}

inline void __attribute__ ((always_inline)) unpack_seq(size_type *it_space, float_type __restrict__ *src, float_type __restrict__ *dst)
{
    size_type buffer_pos = 0;

    for(size_type k=it_space[0]; k<it_space[1]; k++){
        for(size_type j=it_space[2]; j<it_space[3]; j++){
            if(1){
                size_type len = it_space[5]-it_space[4];
                memcpy(dst+k*dimx*dimy + j*dimx + it_space[4], src+buffer_pos, sizeof(float_type)*len);
                buffer_pos+=len;
            } else {
                for(size_type i=it_space[4]; i<it_space[5]; i++){
                    dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos++];
                }
            }
        }
    }
}

int main(int argc, char *argv[]){
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

        size_type **iteration_spaces_pack;              // for each thread, for each nbor, 6 ints
        size_type **iteration_spaces_unpack;              // for each thread, for each nbor, 6 ints

        // explicitly create halo iteration spaces
        iteration_spaces_pack = (size_type**)malloc(sizeof(size_type**)*27);
        iteration_spaces_unpack = (size_type**)malloc(sizeof(size_type**)*27);
        for(int nbid=0; nbid<27; nbid++){
            iteration_spaces_pack[nbid] = (size_type*)malloc(sizeof(size_type*)*6);
            iteration_spaces_unpack[nbid] = (size_type*)malloc(sizeof(size_type*)*6);
        }
        
        Z_RANGE_SRC(iteration_spaces_pack);        
        Z_RANGE_DST(iteration_spaces_unpack);        

        // compute memory footprint
        double n_halo_values = 0;
        for(int nbid=0; nbid<27; nbid++){
            size_type *is = iteration_spaces_unpack[nbid];
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
                    pack_seq(iteration_spaces_pack[nbid], data_cubes[fid], sequential_buffers[rank][nbid][fid]);
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
                    unpack_seq(iteration_spaces_unpack[nbid], sequential_buffers[nb[nbid]][26-nbid][fid], data_cubes[fid]);
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
                    size_type *it_space = iteration_spaces_unpack[nbid];
                    for(size_type k=it_space[0]; k<it_space[1]; k++){
                        for(size_type j=it_space[2]; j<it_space[3]; j++){
                            for(size_type i=it_space[4]; i<it_space[5]; i++){
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
                    pack_seq(iteration_spaces_pack[nbid], data_cubes[fid], sequential_buffers[rank][nbid][fid]);
                    unpack_seq(iteration_spaces_unpack[nbid], sequential_buffers[nb[nbid]][26-nbid][fid], data_cubes[fid]);
                }
            }
        }
#pragma omp barrier
        bytes = n_halo_values*num_fields*num_repetitions*num_ranks*sizeof(float_type)*4;
        if(rank==0) {
            printf("pack+unpack (sequence)\n");
            toc();
        }
    }
}
