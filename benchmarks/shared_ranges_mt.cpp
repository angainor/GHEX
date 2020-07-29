#include "packing_common.h"

/* number of threads X number of cubes X data size */
float_type ***data_cubes;

void __attribute__((noinline))
put_halo(float_type *__restrict__ dst, const float_type *__restrict__ src, size_type *src_space, size_type *dst_space)
{
    size_type ks, kd, js, jd;

    for(ks=src_space[0], kd=dst_space[0]; ks<src_space[1]; ks++, kd++){
        for(js=src_space[2], jd=dst_space[2]; js<src_space[3]; js++, jd++){
            // size_type is, id;
	    // for(is=src_space[4], id=dst_space[4]; is<src_space[5]; is++, id++){
	    //     dst[kd*dimx*dimy + jd*dimx + id] = src[ks*dimx*dimy + js*dimx + is];
            // }
            memcpy(dst + kd*dimx*dimy + jd*dimx + dst_space[4], 
                src + ks*dimx*dimy + js*dimx + src_space[4], 
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
        size_type memsize;

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
            for(size_type k=halo; k<dimz-halo; k++){
                for(size_type j=halo; j<dimy-halo; j++){
                    for(size_type i=halo; i<dimx-halo; i++){
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
