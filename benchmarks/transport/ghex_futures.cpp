#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>
#include <unistd.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>


/* define to use the raw shared message - lower overhead */
#define GHEX_USE_RAW_SHARED_MESSAGE

#include <ghex/common/timer.hpp>
#include <ghex/transport_layer/callback_communicator.hpp>
using MsgType = gridtools::ghex::tl::message_buffer<>;


#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
using FutureType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>::future<void>;
#define USE_CALLBACK_COMM
#else

/* UCX backend */
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;
using FutureType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>::future<void>;

#ifdef USE_UCX_NBR
/* use the GHEX callback framework */
#define USE_CALLBACK_COMM
#else
/* use the UCX's own callback framework */
#include <ghex/transport_layer/ucx/communicator.hpp>
#undef  USE_CALLBACK_COMM
#endif /* USE_UCX_NBR */

#endif /* USE_MPI */


int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;
    gridtools::ghex::timer timer;
    long bytes = 0;
    
#ifdef USE_MPI
    int mode;
#ifdef THREAD_MODE_MULTIPLE
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
	std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
	std::terminate();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif
#endif

    /* TODO this needs to be made per-thread. 
       If we make 'static' variables, then we can't initialize m_rank and anything else
       that used MPI in the constructor, as it will be executed before MPI_Init.
    */
    CommType comm;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);

    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
    
    {
	std::vector<MsgType> msgs;
	FutureType reqs[inflight];
	
	for(int j=0; j<inflight; j++){
	    msgs.push_back(MsgType(buff_size));
	}

	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	int i = 0;
	while(i<niter){
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){

		if(!reqs[j].ready()) continue;

		i++;
		if(rank==0 && (i)%(niter/10)==0) {
		    std::cout << i << " iters\n";		    
		}

		if(rank == 0)
		    reqs[j] = comm.send(1, j, msgs[j]);
		else
		    reqs[j] = comm.recv(0, j, msgs[j]);
		if(i==niter) break;
	    }
	}

	comm.fence();
    }

    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize(); segfault ??
#endif
}
