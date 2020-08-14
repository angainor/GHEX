#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "packing_common.h"

typedef float float_type;

void rank2coord(int rank, int rank_dims[3], int coord[3]);
int coord2rank(int rank_dims[3], int coord[3]);
int get_nbor(int rank_dims[3], int coord[3], int shift, int dim);

void exchange_sendrecv_init(int local_dims[3], int halo);
void exchange_sendrecv(float_type *f, int local_dims[3], int halo,
		       int R_XUP, int R_XDN, int R_YUP, int R_YDN, int R_ZUP, int R_ZDN);

void exchange_pack_init(int local_dims[3], int halo);
void exchange_pack(float_type *f, int local_dims[3], int halo,
		   int R_XUP, int R_XDN, int R_YUP, int R_YDN, int R_ZUP, int R_ZDN);

MPI_Datatype t_xhalo, t_yhalo, t_zhalo;
int rank, size;
float_type *sbuff_l, *sbuff_r, *rbuff_l, *rbuff_r;

int main(int argc, char *argv[]){
    int num_ranks = 1;

    int rank_dims[3] = {0};
    int local_dims[3] = {0};
    int coord[3] = {0};
    int dim = 0;
    int num_repetitions = 0;
    int halo = 0;
    int nfields = 8;
    
    int R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(argc!=7){
	fprintf(stderr, "Usage: <bench name> dim num_repetitions halo rank_space_dimensions (%d)\n", argc);
	exit(1);
    }

    dim = atoi(argv[1]);
    num_repetitions = atoi(argv[2]);
    halo = atoi(argv[3]);   
    
    for(int i=4; i<7; i++){
	rank_dims[i-4] = atoi(argv[i]);
	num_ranks *= rank_dims[i-4];
    }

    if(size != num_ranks){
	fprintf(stderr, "number of ranks does not match the rank space dimension\n");
	exit(1);
    }
    
    /* make a cartesian thread space */
    if(0 ==  rank) printf("cart rank_dims %d %d %d\n", rank_dims[0], rank_dims[1], rank_dims[2]);
   
    float_type **fields;
    size_type memsize;
    size_type dimx = (dim + 2*halo);
    size_type dimy = (dim + 2*halo);
    size_type dimz = (dim + 2*halo);
    size_type k, j, i;

    local_dims[0] = dimx;
    local_dims[1] = dimy;
    local_dims[2] = dimz;
    
    memsize = sizeof(float_type)*dimx*dimy*dimz;

    fields = (float_type**)malloc(sizeof(float_type*)*nfields);
    for(int fi=0; fi<nfields; fi++){
	void **ptr = (void**)&fields[fi];
	int retval = posix_memalign(ptr, 4096, memsize);
	(void)retval;
	memset(fields[fi], 0, memsize);
	for(k=halo; k<dimz-halo; k++){
	    for(j=halo; j<dimy-halo; j++){
		for(i=halo; i<dimx-halo; i++){
		    fields[fi][k*dimx*dimy + j*dimx + i] = rank+1;
		}
	    }
	}
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* find coords in cartesian rank space */
    rank2coord(rank, rank_dims, coord);
    // printf("%d == %d : %d %d %d\n", rank, coord2rank(rank_dims, coord), coord[0], coord[1], coord[2]);
    
    /* find neighbor ranks */
    R_XUP = get_nbor(rank_dims, coord, +1, 1);
    R_XDN = get_nbor(rank_dims, coord, -1, 1);
    R_YUP = get_nbor(rank_dims, coord, +1, 2);
    R_YDN = get_nbor(rank_dims, coord, -1, 2);
    R_ZUP = get_nbor(rank_dims, coord, +1, 3);
    R_ZDN = get_nbor(rank_dims, coord, -1, 3);    
    // printf("rank %d: %d %d %d %d %d %d\n", rank, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN);

    /* sendrecv implementation */
    exchange_sendrecv_init(local_dims, halo);
    for(int fid=0; fid<num_fields; fid++){
	exchange_sendrecv(fields[fid], local_dims, halo, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN);	
    }

    tic();
    for(int it=0; it<num_repetitions; it++){
	for(int fid=0; fid<num_fields; fid++){
	    exchange_sendrecv(fields[fid], local_dims, halo, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN);	
	}
    }
    bytes = local_dims[0]*local_dims[1]*halo*6;
    bytes = sizeof(float_type)*bytes*num_fields*num_repetitions*num_ranks*2;
    if(rank==0) {
	printf("exchange_sendrecv\n");
	toc();
    }   

    /* pack, isend/irecv, unpack */
    exchange_pack_init(local_dims, halo);
    for(int fid=0; fid<num_fields; fid++){
	exchange_pack(fields[fid], local_dims, halo, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN);
    }

    tic();
    for(int it=0; it<num_repetitions; it++){
	for(int fid=0; fid<num_fields; fid++){
	    exchange_pack(fields[fid], local_dims, halo, R_XUP, R_XDN, R_YUP, R_YDN, R_ZUP, R_ZDN);
	}
    }
    bytes = local_dims[0]*local_dims[1]*halo*6;
    bytes = sizeof(float_type)*bytes*num_fields*num_repetitions*num_ranks*2;
    if(rank==0) {
	printf("exchange_pack\n");
	toc();
    }   
    
    // if(rank==0) PRINT_CUBE(fields[0]);   
    
    MPI_Finalize();
}

void comm_x_pack(float_type *f, int local_dims[3], int halo, int RUP, int RDN) {

    /* TODO: assumed all local dimensions are equal */
    size_t memsize = sizeof(float_type)*local_dims[0]*local_dims[1]*halo;
    int position = 0;
    MPI_Request requests[4];

    position = 0;
    MPI_Pack(f+local_dims[0]-2*halo, 1, t_xhalo,
	     sbuff_r, memsize, &position, MPI_COMM_WORLD);
    position = 0;
    MPI_Pack(f+halo, 1, t_xhalo,
	     sbuff_l, memsize, &position, MPI_COMM_WORLD);

    MPI_Irecv(rbuff_r, memsize, MPI_PACKED, RUP, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(rbuff_l, memsize, MPI_PACKED, RDN, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(sbuff_r, memsize, MPI_PACKED, RUP, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(sbuff_l, memsize, MPI_PACKED, RDN, 1, MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    position = 0;
    MPI_Unpack(rbuff_l, memsize, &position, f, 1, t_xhalo, MPI_COMM_WORLD);
    position = 0;
    MPI_Unpack(rbuff_r, memsize, &position, f+local_dims[0]-halo, 1, t_xhalo, MPI_COMM_WORLD);
}

void comm_y_pack(float_type *f, int local_dims[3], int halo, int RUP, int RDN) {

    /* TODO: assumed all local dimensions are equal */
    size_t memsize = sizeof(float_type)*local_dims[0]*local_dims[1]*halo;
    int position = 0;
    MPI_Request requests[4];

    position = 0;
    MPI_Pack(f+local_dims[0]*(local_dims[1]-2*halo), 1, t_yhalo,
	     sbuff_r, memsize, &position, MPI_COMM_WORLD);
    position = 0;
    MPI_Pack(f+local_dims[0]*halo, 1, t_yhalo,
	     sbuff_l, memsize, &position, MPI_COMM_WORLD);

    MPI_Irecv(rbuff_r, memsize, MPI_PACKED, RUP, 2, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(rbuff_l, memsize, MPI_PACKED, RDN, 2, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(sbuff_r, memsize, MPI_PACKED, RUP, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(sbuff_l, memsize, MPI_PACKED, RDN, 2, MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    position = 0;
    MPI_Unpack(rbuff_l, memsize, &position, f, 1, t_yhalo, MPI_COMM_WORLD);
    position = 0;
    MPI_Unpack(rbuff_r, memsize, &position, f+local_dims[0]*(local_dims[1]-halo), 1, t_yhalo, MPI_COMM_WORLD);
}

void comm_z_pack(float_type *f, int local_dims[3], int halo, int RUP, int RDN) {

    /* TODO: assumed all local dimensions are equal */
    size_t memsize = sizeof(float_type)*local_dims[0]*local_dims[1]*halo;
    MPI_Request requests[4];

    MPI_Irecv(f+local_dims[0]*local_dims[1]*(local_dims[2]-halo), memsize, MPI_PACKED, RUP, 3, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(f, memsize, MPI_PACKED, RDN, 3, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(f+local_dims[0]*local_dims[1]*(local_dims[2]-2*halo), memsize, MPI_PACKED, RUP, 3, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(f+local_dims[0]*local_dims[1]*halo, memsize, MPI_PACKED, RDN, 3, MPI_COMM_WORLD, &requests[3]);
    
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

void exchange_pack_init(int local_dims[3], int halo) {
    /* TODO: assumed all local dimensions are equal */
    size_t memsize = sizeof(float_type)*local_dims[0]*local_dims[1]*halo;
    void **ptr;
    int retval;

    /* need the types */
    exchange_sendrecv_init(local_dims, halo);
    
    ptr = (void**)&sbuff_l;
    retval = posix_memalign(ptr, 4096, memsize);
    memset(sbuff_l, 0, memsize);
    ptr = (void**)&sbuff_r;
    retval = posix_memalign(ptr, 4096, memsize);
    memset(sbuff_r, 0, memsize);

    ptr = (void**)&rbuff_l;
    retval = posix_memalign(ptr, 4096, memsize);
    memset(rbuff_l, 0, memsize);
    ptr = (void**)&rbuff_r;
    retval = posix_memalign(ptr, 4096, memsize);
    memset(rbuff_r, 0, memsize);

    (void)retval;
}

void exchange_pack(float_type *f, int local_dims[3], int halo,
		   int R_XUP, int R_XDN, int R_YUP, int R_YDN, int R_ZUP, int R_ZDN) {
    comm_x_pack(f, local_dims, halo, R_XUP, R_XDN);
    comm_y_pack(f, local_dims, halo, R_YUP, R_YDN);
    comm_z_pack(f, local_dims, halo, R_ZUP, R_ZDN);
}

void comm_x_sendrecv(float_type *f, int local_dims[3], int halo, int RUP, int RDN) {
    MPI_Sendrecv(f+local_dims[0]-2*halo, 1, t_xhalo, RUP, 1,
    		 f, 1, t_xhalo, RDN, 1,
    		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f+halo, 1, t_xhalo, RDN, 2,
    		 f+local_dims[0]-halo, 1, t_xhalo, RUP, 2,
    		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void comm_y_sendrecv(float_type *f, int local_dims[3], int halo, int RUP, int RDN) {
    MPI_Sendrecv(f+local_dims[0]*(local_dims[1]-2*halo), 1, t_yhalo, RUP, 3,
    		 f, 1, t_yhalo, RDN, 3,
    		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f+local_dims[0]*halo, 1, t_yhalo, RDN, 4,
    		 f+local_dims[0]*(local_dims[1]-halo), 1, t_yhalo, RUP, 4,
    		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void comm_z_sendrecv(float_type *f, int local_dims[3], int halo, int RUP, int RDN) {
    MPI_Sendrecv(f+local_dims[0]*local_dims[1]*(local_dims[2]-2*halo), 1, t_zhalo, RUP, 5,
    		 f, 1, t_zhalo, RDN, 5,
    		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(f+local_dims[0]*local_dims[1]*halo, 1, t_zhalo, RDN, 6,
    		 f+local_dims[0]*local_dims[1]*(local_dims[2]-halo), 1, t_zhalo, RUP, 6,
    		 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void exchange_sendrecv_init(int local_dims[3], int halo) {

    /* create halo types: vector */
    MPI_Type_vector(local_dims[0]*local_dims[2], halo, local_dims[0], MPI_REAL4, &t_xhalo);
    MPI_Type_commit(&t_xhalo);
    MPI_Type_vector(local_dims[2], halo*local_dims[0], local_dims[0]*local_dims[1], MPI_REAL4, &t_yhalo);
    MPI_Type_commit(&t_yhalo);
    MPI_Type_vector(1, halo*local_dims[0]*local_dims[1], 1, MPI_REAL4, &t_zhalo);
    MPI_Type_commit(&t_zhalo);
}

void exchange_sendrecv(float_type *f, int local_dims[3], int halo,
		       int R_XUP, int R_XDN, int R_YUP, int R_YDN, int R_ZUP, int R_ZDN) {
    comm_x_sendrecv(f, local_dims, halo, R_XUP, R_XDN);
    comm_y_sendrecv(f, local_dims, halo, R_YUP, R_YDN);
    comm_z_sendrecv(f, local_dims, halo, R_ZUP, R_ZDN);
}

void rank2coord(int rank, int rank_dims[3], int coord[3]) {
    int tmp;
    tmp = rank;             coord[0] = tmp % rank_dims[0];
    tmp = tmp/rank_dims[0]; coord[1] = tmp % rank_dims[1];
    tmp = tmp/rank_dims[1]; coord[2] = tmp;
}

int coord2rank(int rank_dims[3], int coord[3]){
    return (coord[2]*rank_dims[1] + coord[1])*rank_dims[0] + coord[0];
}

int get_nbor(int rank_dims[3], int coord[3], int shift, int dim){
    int nbor;
    int coord_old = coord[dim-1];
    coord[dim-1] = coord[dim-1] + shift;
    if (coord[dim-1] == rank_dims[dim-1]) coord[dim-1] = 0;
    if (coord[dim-1] < 0) coord[dim-1] = rank_dims[dim-1]-1;
    nbor = coord2rank(rank_dims, coord);
    coord[dim-1] = coord_old;
    return nbor;
}
