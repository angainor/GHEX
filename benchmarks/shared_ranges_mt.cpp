#include "packing_common.h"
#include <omp.h>

int dims[3] = {4, 4, 2};
size_t dimx, dimy, dimz;

/* number of threads X number of cubes X data size */
float_type ***data_cubes;

inline void put_halo(float_type *__restrict__ dst, const float_type *__restrict__ src, int *src_space, int *dst_space)
{
    int ks, kd, js, jd, is, id;
    for(ks=src_space[0], kd=dst_space[0]; ks<src_space[1]; ks++, kd++){
        for(js=src_space[2], jd=dst_space[2]; js<src_space[3]; js++, jd++){
	    for(is=src_space[4], id=dst_space[4]; is<src_space[5]; is++, id++){
		dst[kd*dimx*dimy + jd*dimx + id] = src[ks*dimx*dimy + js*dimx + is];
            }
        }
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

#define Y_BLOCK(XBL, rank, coords, arg, k, nbz, id)             \
    {                                                           \
        int j;                                                  \
        int nby = -1;						\
        for(j=halo; j<2*halo; j++){                             \
            XBL(rank, coords, arg, k, nbz, j, nby, id);         \
        }                                                       \
        nby = 0;                                                \
        for(j=halo; j<halo + local_dims[1]; j++){               \
            XBL(rank, coords, arg, k, nbz, j, nby, id);         \
        }                                                       \
        nby = 1;                                                \
        for(j=local_dims[1]; j<halo + local_dims[1]; j++){      \
            XBL(rank, coords, arg, k, nbz, j, nby, id);         \
        }                                                       \
    }

#define Z_BLOCK(XBL, rank, coords, arg, id)                     \
    {                                                           \
        int k;                                                  \
        int nbz = -1;						\
        for(k=halo; k<2*halo; k++){                             \
            Y_BLOCK(XBL, rank, coords, arg, k, nbz, id);        \
        }                                                       \
        nbz = 0;                                                \
        for(k=halo; k<halo + local_dims[2]; k++){               \
            Y_BLOCK(XBL, rank, coords, arg, k, nbz, id);        \
        }                                                       \
        nbz = 1;                                                \
        for(k=local_dims[2]; k<halo + local_dims[2]; k++){      \
            Y_BLOCK(XBL, rank, coords, arg, k, nbz, id);        \
        }                                                       \
    }

#define FILL_IT_SPACE()                         \
    it_space[nbid][0] = zlo;                    \
    it_space[nbid][1] = zhi;                    \
    it_space[nbid][2] = ylo;                    \
    it_space[nbid][3] = yhi;                    \
    it_space[nbid][4] = xlo;                    \
    it_space[nbid][5] = xhi;                    \

inline void X_RANGE_SRC(int **it_space, int zlo, int zhi, int nbz, int ylo, int yhi, int nby)
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

#define Y_RANGE_SRC(it_space, zlo, zhi, nbz)				\
    {                                                                   \
        X_RANGE_SRC(it_space, zlo, zhi, nbz, halo, 2*halo, -1);		\
        X_RANGE_SRC(it_space, zlo, zhi, nbz, halo, halo + local_dims[1], 0); \
        X_RANGE_SRC(it_space, zlo, zhi, nbz, local_dims[1], halo + local_dims[1], 1); \
    }

#define Z_RANGE_SRC(it_space)						\
    {                                                                   \
        Y_RANGE_SRC(it_space, halo, 2*halo, -1);			\
        Y_RANGE_SRC(it_space, halo, halo + local_dims[2], 0);		\
        Y_RANGE_SRC(it_space, local_dims[2], halo + local_dims[2], 1);	\
    }

inline void X_RANGE_DST(int **it_space, int zlo, int zhi, int nbz, int ylo, int yhi, int nby)
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


int main()
{    
    int num_ranks = 1;

#pragma omp parallel
    {
#pragma omp master
        num_ranks = omp_get_num_threads();
    }

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

        int **iteration_spaces_pack;              // for each thread, for each nbor, 6 ints
        int **iteration_spaces_unpack;              // for each thread, for each nbor, 6 ints

        // explicitly create halo iteration spaces
        iteration_spaces_pack = (int**)malloc(sizeof(int**)*27);
        iteration_spaces_unpack = (int**)malloc(sizeof(int**)*27);
        for(int nbid=0; nbid<27; nbid++){
            iteration_spaces_pack[nbid] = (int*)malloc(sizeof(int*)*6);
            iteration_spaces_unpack[nbid] = (int*)malloc(sizeof(int*)*6);
        }
        
        Z_RANGE_SRC(iteration_spaces_pack);        
        Z_RANGE_DST(iteration_spaces_unpack);        

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
	
        /* compute data size */
        dimx = (local_dims[0] + 2*halo);
        dimy = (local_dims[1] + 2*halo);
        dimz = (local_dims[2] + 2*halo);
        memsize = sizeof(float_type)*dimx*dimy*dimz;

#pragma omp barrier

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
        for(int it=0; it<2; it++){
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    if(nbid==13) continue;
		    float_type *src = data_cubes[rank][fid];
		    float_type *dst = data_cubes[nb[nbid]][fid];
                    put_halo(dst, src, iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                }
            }
	}

#pragma omp barrier
        tic();
        for(int it=0; it<num_repetitions; it++){
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    if(nbid==13) continue;
		    float_type *src = data_cubes[rank][fid];
		    float_type *dst = data_cubes[nb[nbid]][fid];
                    put_halo(dst, src, iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                }
            }
#pragma omp barrier
        }
#pragma omp barrier
        bytes = 2*num_repetitions*num_ranks*num_fields*(dimx*dimy*dimz-local_dims[0]*local_dims[1]*local_dims[2])*sizeof(float_type);
        if(rank==0) {
            printf("put_halo\n");
            toc();
        }

	// if(rank==0){
	//     PRINT_CUBE(data_cubes[rank][0]);
	// }
	
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
