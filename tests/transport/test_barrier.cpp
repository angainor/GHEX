#include <ghex/threads/none/primitives.hpp>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

using threading = gridtools::ghex::threads::none::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

TEST(transport, barrier) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    for(int i=0; i<10000; i++)  {
        comm.barrier();
    }

    EXPECT_FALSE(comm.progress());
}