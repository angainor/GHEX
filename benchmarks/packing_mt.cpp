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

// for double copy communication
float_type **compact_buffers_cpy;     // a large buffer to store all fields, for each neighbor, including self
float_type ***sequential_buffers_cpy; // num_fields small buffers for each neighbor, including self
#pragma omp threadprivate(compact_buffers_cpy, sequential_buffers_cpy)

inline void __attribute__ ((always_inline)) x_pack_seq(const int rank, const int [3],
    int *, int k, int nbz, int j, int nby, int id)
{
    int nbid;
    float *dst, *src;


    // which cube to pack
    src = data_cubes[id];

    nbid = id2nbid(-1, nby, nbz);
    dst = sequential_buffers[rank][nbid][id];

#ifdef USE_MEMCPY
    memcpy(dst+buffer_pos[nbid], src+k*dimx*dimy + j*dimx + halo, sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=halo; i<2*halo; i++){
        dst[buffer_pos[nbid]++] = src[k*dimx*dimy + j*dimx + i];
    }
#endif

    nbid = id2nbid(0, nby, nbz);
    if(nbid != 13) {
        dst = sequential_buffers[rank][nbid][id];

#ifdef USE_MEMCPY
        memcpy(dst+buffer_pos[nbid], src+k*dimx*dimy + j*dimx + halo, sizeof(float_type)*(local_dims[0]));
        buffer_pos[nbid]+=local_dims[0];
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for(int i=halo; i<halo + local_dims[0]; i++){
            dst[buffer_pos[nbid]++] = src[k*dimx*dimy + j*dimx + i];
        }
#endif
    }

    nbid = id2nbid(1, nby, nbz);
    dst = sequential_buffers[rank][nbid][id];

#ifdef USE_MEMCPY
    memcpy(dst+buffer_pos[nbid], src+k*dimx*dimy + j*dimx + local_dims[0], sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=local_dims[0]; i<halo + local_dims[0]; i++){
        dst[buffer_pos[nbid]++] = src[k*dimx*dimy + j*dimx + i];
    }
#endif

}

inline void __attribute__ ((always_inline)) x_pack_compact(const int rank, const int [3],
    int *, int k, int nbz, int j, int nby, int id)
{
    int nbid;
    float *dst, *src;

    // which cube to pack
    src = data_cubes[id];

    nbid = id2nbid(-1, nby, nbz);
    dst = compact_buffers[rank][nbid];

#ifdef USE_MEMCPY
    memcpy(dst+buffer_pos[nbid], src+k*dimx*dimy + j*dimx + halo, sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=halo; i<2*halo; i++){
        dst[buffer_pos[nbid]++] = src[k*dimx*dimy + j*dimx + i];
    }
#endif

    nbid = id2nbid(0, nby, nbz);
    if(nbid != 13) {
        dst = compact_buffers[rank][nbid];

#ifdef USE_MEMCPY
        memcpy(dst+buffer_pos[nbid], src+k*dimx*dimy + j*dimx + halo, sizeof(float_type)*(local_dims[0]));
        buffer_pos[nbid]+=local_dims[0];
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for(int i=halo; i<halo + local_dims[0]; i++){
            dst[buffer_pos[nbid]++] = src[k*dimx*dimy + j*dimx + i];
        }
#endif
    }

    nbid = id2nbid(1, nby, nbz);
    dst = compact_buffers[rank][nbid];

#ifdef USE_MEMCPY
    memcpy(dst+buffer_pos[nbid], src+k*dimx*dimy + j*dimx + local_dims[0], sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=local_dims[0]; i<halo + local_dims[0]; i++){
        dst[buffer_pos[nbid]++] = src[k*dimx*dimy + j*dimx + i];
    }
#endif
}

inline void __attribute__ ((always_inline)) x_unpack_seq(const int, const int coords[3],
    int *, int k, int nbz, int j, int nby, int id)
{
    int nb, nbid;
    float *dst, *src;

    // which cube to unpack
    dst = data_cubes[id];

    // flip signs to get our nbid in our neighbors context: where did he pack our data?
    nbid = id2nbid(1, nby, nbz);
    nb   = coord2rank(dims, coords[0]-1, coords[1]-nby, coords[2]-nbz);
    src  = sequential_buffers[nb][nbid][id];

#ifdef USE_MEMCPY
    memcpy(dst+k*dimx*dimy + j*dimx, src+buffer_pos[nbid], sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=0; i<halo; i++){
        dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos[nbid]++];
    }
#endif

    nbid = id2nbid(0, nby, nbz);
    if(nbid != 13) {
        nb   = coord2rank(dims, coords[0], coords[1]-nby, coords[2]-nbz);
        src = sequential_buffers[nb][nbid][id];

#ifdef USE_MEMCPY
        memcpy(dst+k*dimx*dimy + j*dimx + halo, src+buffer_pos[nbid], sizeof(float_type)*local_dims[0]);
        buffer_pos[nbid]+=local_dims[0];
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for(int i=halo; i<halo + local_dims[0]; i++){
            dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos[nbid]++];
        }
#endif
    }

    nbid = id2nbid(-1, nby, nbz);
    nb   = coord2rank(dims, coords[0]+1, coords[1]-nby, coords[2]-nbz);
    src = sequential_buffers[nb][nbid][id];

#ifdef USE_MEMCPY
    memcpy(dst+k*dimx*dimy + j*dimx + local_dims[0]+halo, src+buffer_pos[nbid], sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=local_dims[0]+halo; i<2*halo + local_dims[0]; i++){
        dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos[nbid]++];
    }
#endif
}

inline void __attribute__ ((always_inline)) x_unpack_compact(const int, const int coords[3],
    int *, int k, int nbz, int j, int nby, int id)
{
    int nb, nbid;
    float *dst, *src;

    // which cube to unpack
    dst = data_cubes[id];

    // flip signs to get our nbid in our neighbors context: where did he pack our data?
    nbid = id2nbid(1, nby, nbz);
    nb   = coord2rank(dims, coords[0]-1, coords[1]-nby, coords[2]-nbz);
    src  = compact_buffers[nb][nbid];

#ifdef USE_MEMCPY
    memcpy(dst+k*dimx*dimy + j*dimx, src+buffer_pos[nbid], sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=0; i<halo; i++){
        dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos[nbid]++];
    }
#endif

    nbid = id2nbid(0, nby, nbz);
    if(nbid != 13) {
        nb   = coord2rank(dims, coords[0], coords[1]-nby, coords[2]-nbz);
        src  = compact_buffers[nb][nbid];

#ifdef USE_MEMCPY
        memcpy(dst+k*dimx*dimy + j*dimx + halo, src+buffer_pos[nbid], sizeof(float_type)*local_dims[0]);
        buffer_pos[nbid]+=local_dims[0];
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for(int i=halo; i<halo + local_dims[0]; i++){
            dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos[nbid]++];
        }
#endif
    }

    nbid = id2nbid(-1, nby, nbz);
    nb   = coord2rank(dims, coords[0]+1, coords[1]-nby, coords[2]-nbz);
    src  = compact_buffers[nb][nbid];

#ifdef USE_MEMCPY
    memcpy(dst+k*dimx*dimy + j*dimx + local_dims[0]+halo, src+buffer_pos[nbid], sizeof(float_type)*halo);
    buffer_pos[nbid]+=halo;
#else
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for(int i=local_dims[0]+halo; i<2*halo + local_dims[0]; i++){
        dst[k*dimx*dimy + j*dimx + i] = src[buffer_pos[nbid]++];
    }
#endif
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

        // compute local neighbor ids: nbid and nb
        int nb[27] = {};
        int nbid[27] = {};
        int local_nbid = 0;
        for(int nbz=-1; nbz<=1; nbz++){
            for(int nby=-1; nby<=1; nby++){
                for(int nbx=-1; nbx<=1; nbx++){
                    nbid[local_nbid] = id2nbid(nbx, nby, nbz);
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

        // allocate the halo copy buffers
        compact_buffers_cpy = (float_type**)calloc(sizeof(float_type**), 27);
        for(int i=0; i<27; i++){
            ptr = (void**)(compact_buffers_cpy+i);
            posix_memalign(ptr, 4096, halosize*num_fields);
            memset(compact_buffers_cpy[i], 0, halosize*num_fields);
        }
        sequential_buffers_cpy = (float_type***)calloc(sizeof(float_type***), 27);
        for(int j=0; j<27; j++){
            sequential_buffers_cpy[j] = (float_type**)calloc(sizeof(float_type**), num_fields);
            for(int i=0; i<num_fields; i++){
                ptr = (void**)(sequential_buffers_cpy[j]+i);
                posix_memalign(ptr, 4096, halosize);
                memset(sequential_buffers_cpy[j][i], 0, halosize);
            }
        }

        // packing loop: put all data into split comm buffers - separate for each field
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(int fid=0; fid<num_fields; fid++){
                memset(buffer_pos, 0, sizeof(int)*27);
                Z_BLOCK(x_pack_seq, rank, coords, NULL, fid);
            }
        }
#pragma omp barrier
        bytes = 0;
        for(int j=0; j<27; j++) bytes += buffer_pos[j];
        bytes = bytes*num_fields*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("packing (sequence)\n");
            toc();
        }

        // validate
        for(int r=0; r<num_ranks; r++){
            if(rank==r){
                for(int fid=0; fid<num_fields; fid++){
                    for(int i=0; i<27; i++){
                        if(i!=13)
                            for(int j=0; j<buffer_pos[i]; j++)
                                if(sequential_buffers[rank][i][fid][j]!=rank+1) printf(".");
                    }
                }
            }
#pragma omp barrier
        }

        // copy comm buffers from the neighbors
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(local_nbid = 0; local_nbid<27; ){
                if(local_nbid!=13)
                    for(int fid=0; fid<num_fields; fid++){
                        memcpy(sequential_buffers_cpy[local_nbid][fid], sequential_buffers[nb[local_nbid]][nbid[local_nbid]][fid], 
                            sizeof(float_type)*buffer_pos[local_nbid]);
                    }
                local_nbid++;
            }
        }
#pragma omp barrier
        bytes = 0;
        for(int j=0; j<27; j++) bytes += buffer_pos[j];
        bytes = bytes*num_fields*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("copy (sequence)\n");
            toc();
        }

        // unpacking loop
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(int fid=0; fid<num_fields; fid++){
                memset(buffer_pos, 0, sizeof(int)*27);
                Z_BLOCK_UNPACK(x_unpack_seq, rank, coords, NULL, fid);
            }
        }
#pragma omp barrier
        bytes = 0;
        for(int j=0; j<27; j++) bytes += buffer_pos[j];
        bytes = bytes*num_fields*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("unpacking (sequence)\n");
            toc();
        }

        // validate
        for(int r=0; r<num_ranks; r++){
            if(rank==r){
                for(int fid=0; fid<num_fields; fid++){
                    for(int i=0; i<27; i++){
                        if(i!=13)
                            for(int j=0; j<buffer_pos[i]; j++)
                                if(sequential_buffers[rank][i][fid][j]!=rank+1) printf(".");
                    }
                }
            }
#pragma omp barrier
        }

        // packing loop: put all data into compact comm buffers - one per neighbor, fields combined
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            memset(buffer_pos, 0, sizeof(int)*27);
            for(int fid=0; fid<num_fields; fid++){
                Z_BLOCK(x_pack_compact, rank, coords, NULL, fid);
            }
        }
#pragma omp barrier
        bytes = 0;
        for(int j=0; j<27; j++) bytes += buffer_pos[j];
        bytes = bytes*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("packing (compact)\n");
            toc();
        }

        // validate
        for(int r=0; r<1; r++){
            if(rank==r){
                for(int i=0; i<27; i++){
                    if(i!=13)
                        for(int j=0; j<buffer_pos[i]; j++)
                            if(compact_buffers[rank][i][j] != rank+1) printf(".");
                }
            }
#pragma omp barrier
        }


        // copy comm buffers from the neighbors
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(local_nbid = 0; local_nbid<27; local_nbid++){
                memcpy(compact_buffers_cpy[local_nbid], compact_buffers[nb[local_nbid]][nbid[local_nbid]], 
                    sizeof(float_type)*buffer_pos[local_nbid]);
            }
        }
#pragma omp barrier
        bytes = 0;
        for(int j=0; j<27; j++) bytes += buffer_pos[j];
        bytes = bytes*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("copy (compact)\n");
            toc();
        }

        // unpacking loop
#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            memset(buffer_pos, 0, sizeof(int)*27);
            for(int fid=0; fid<num_fields; fid++){
                Z_BLOCK(x_unpack_compact, rank, coords, NULL, fid);
            }
        }
#pragma omp barrier
        bytes = 0;
        for(int j=0; j<27; j++) bytes += buffer_pos[j];
        bytes = bytes*num_repetitions*num_ranks*sizeof(float_type)*2;
        if(rank==0) {
            printf("unpacking (compact)\n");
            toc();
        }
    }
}
