#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int dims[3] = {0}, periodic[3] = {1,1,1}, coords[3], nb_cart;
    int size, rank, rank_cart;
    MPI_Comm comm_cart;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Dims_create(size, 3, dims);
    if(rank==0) printf("dims %d %d %d\n", dims[0], dims[1], dims[2]);

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, 1, &comm_cart);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_cart);
    MPI_Cart_coords(comm_cart, rank_cart, 3, coords);
    coords[0] += 1;
    MPI_Cart_rank(comm_cart, coords, &nb_cart);
    printf("%d %d nb %d\n", rank, rank_cart, nb_cart);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
