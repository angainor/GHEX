#include "packing_common.h"
#include <omp.h>
#include <stdint.h>

int dims[3] = {4, 4, 2};
size_t dimx, dimy, dimz;

/* number of threads X number of cubes X data size */
float_type ***data_cubes;

typedef int_fast64_t size_type;
typedef unsigned char byte;

void __attribute__((noinline))
put_halo(float_type *__restrict__ dst, const float_type *__restrict__ src, size_type *src_space, size_type *dst_space)
{
    size_type ks, kd, js, jd, is, id;

    for(ks=src_space[0], kd=dst_space[0]; ks<src_space[1]; ks++, kd++){
        for(js=src_space[2], jd=dst_space[2]; js<src_space[3]; js++, jd++){
	    for(size_type is=src_space[4], id=dst_space[4]; is<src_space[5]; is++, id++){
	        dst[kd*dimx*dimy + jd*dimx + id] = src[ks*dimx*dimy + js*dimx + is];
            }
            // memcpy(dst + kd*dimx*dimy + jd*dimx + dst_space[4], 
            //     src + ks*dimx*dimy + js*dimx + src_space[4], 
            //     sizeof(float_type)*(src_space[5]-src_space[4]));
        }
    }
}

byte*  __attribute__((noinline)) get_buffer3(float_type *m_buffer, const size_type* coord, const size_type* inner_offset, const size_type *outer_stride)
{
    return reinterpret_cast<byte*>(m_buffer) +
        (coord[0] + inner_offset[0]) * outer_stride[0] +
        (coord[1] + inner_offset[1]) * outer_stride[1] +
        (coord[2] + inner_offset[2]) * outer_stride[2] ;
}


void __attribute__((noinline))
put_halo2(float_type *__restrict__ dst, float_type *__restrict__ src, size_type *src_space, size_type *dst_space,
    const size_type* inner_offset, const size_type * outer_stride)
{
    size_type ks, kd, js, jd, is, id;
    byte *dptr, *sptr;
    size_type coord[3];

    for(ks=src_space[0], kd=dst_space[0]; ks<src_space[1]; ks++, kd++){
        for(js=src_space[2], jd=dst_space[2]; js<src_space[3]; js++, jd++){
            {
                const size_type coord[3]{0,jd,kd};
                dptr = get_buffer3(dst, coord, inner_offset, outer_stride);
            }
            {
                const size_type coord[3]{0,js,ks};
                sptr = get_buffer3(src, coord, inner_offset, outer_stride);
            }

	    // for(size_type is=src_space[4]*sizeof(float_type), id=dst_space[4]*sizeof(float_type);  is<src_space[5]*sizeof(float_type); is++, id++){
	    //     dptr[id] = sptr[is];
            // }
            memcpy(dptr+dst_space[4]*sizeof(float_type), 
                sptr+src_space[4]*sizeof(float_type), 
                sizeof(float_type)*(src_space[5]-src_space[4]));
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

inline void X_RANGE_SRC(size_type **it_space, size_type zlo, size_type zhi, size_type nbz, size_type ylo, size_type yhi, size_type nby)
{
    size_type xlo, xhi, nbx, nbid;
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

        size_type **iteration_spaces_pack;              // for each thread, for each nbor, 6 size_types
        size_type **iteration_spaces_unpack;              // for each thread, for each nbor, 6 ints

        // explicitly create halo iteration spaces
        iteration_spaces_pack = (size_type**)malloc(sizeof(size_type**)*27);
        iteration_spaces_unpack = (size_type**)malloc(sizeof(size_type**)*27);
        for(size_type nbid=0; nbid<27; nbid++){
            iteration_spaces_pack[nbid] = (size_type*)malloc(sizeof(size_type*)*6);
            iteration_spaces_unpack[nbid] = (size_type*)malloc(sizeof(size_type*)*6);
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
	
        size_type inner_offset[3]{0,0,0};
        size_type outer_stride[3]{1*sizeof(float_type),dimx*sizeof(float_type),dimx*dimy*sizeof(float_type)};

#pragma omp barrier
        /* warmup */
        for(int it=0; it<2; it++){
            for(int fid=0; fid<num_fields; fid++){
                for(int nbid=0; nbid<27; nbid++){
                    if(nbid==13) continue;
		    float_type *src = data_cubes[rank][fid];
		    float_type *dst = data_cubes[nb[nbid]][fid];
                    put_halo2(dst, src, iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid], inner_offset, outer_stride);
                    // put_halo(dst, src, iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
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
                    put_halo2(dst, src, iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid], inner_offset, outer_stride);
                    //put_halo(dst, src, iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                }
                // int nbid;
                // nbid=3; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=9; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=10; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=20; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=7; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=2; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=15; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=4; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=25; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=6; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=21; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=23; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=8; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=18; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=5; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=14; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=16; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=17; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=12; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=1; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=0; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=22; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=26; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=24; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=11; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
                // nbid=19; put_halo(data_cubes[nb[nbid]][fid], data_cubes[rank][fid], iteration_spaces_pack[nbid], iteration_spaces_unpack[nbid]);
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
