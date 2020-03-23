/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */

#include <array>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include <ghex/communication_object_2.hpp>
#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/domain_descriptor.hpp>
#include <ghex/structured/simple_field_wrapper.hpp>
#include <ghex/transport_layer/mpi/context.hpp>
#include <ghex/threads/none/primitives.hpp>
#include <ghex/common/timer.hpp>

using timer_type = gridtools::ghex::timer;

using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::none::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;
using arch_type = gridtools::ghex::cpu;
using domain_descriptor_type = gridtools::ghex::structured::domain_descriptor<int,3>;

using float_type = float;
const std::array<int,3> local_dims = {64, 64, 64};
const int halo = 5;
const int num_fields = 8;
const int num_repetitions = 100;

const std::array<int,3> local_dims_extended = {local_dims[0]+2*halo, local_dims[1]+2*halo, local_dims[2]+2*halo};
const std::array<int,3> offset = {halo,halo,halo};
const std::size_t num_grid_points = (std::size_t)(local_dims_extended[0])*(std::size_t)(local_dims_extended[1])*(std::size_t)(local_dims_extended[2]);
const std::array<int,6> halo_vec{halo,halo,halo,halo,halo,halo};

template<typename Context, typename Communicator, typename Pattern, typename Fields>
void run_compact(Context& context, Communicator comm, Pattern& pattern, Fields& fields) {
    // communication object
    auto co = gridtools::ghex::make_communication_object<Pattern>(comm);
    timer_type timer;
    // exchange
    co.pack_unpack(
        pattern(fields[0]),
        pattern(fields[1]),
        pattern(fields[2]),
        pattern(fields[3]),
        pattern(fields[4]),
        pattern(fields[5]),
        pattern(fields[6]),
        pattern(fields[7]));
    for (int i=0; i<num_repetitions; ++i)
    {
        timer.tic();
        co.pack_unpack(
            pattern(fields[0]),
            pattern(fields[1]),
            pattern(fields[2]),
            pattern(fields[3]),
            pattern(fields[4]),
            pattern(fields[5]),
            pattern(fields[6]),
            pattern(fields[7]));
        timer.toc();
    }
    if (comm.rank() == 0){
        std::cout << "rank 0:    mean exchange time compact:                  " << timer.mean()/1000
                  << " ± " << timer.stddev()/1000 << " ms" << std::endl;
        std::cout << "rank 0: init  " << co.tinit.mean()/1000 << std::endl;
        std::cout << "rank 0: pack  " << co.tpack.mean()/1000 << std::endl;
        std::cout << "rank 0: upack " << co.tupack.mean()/1000 << std::endl;
    }
    auto global_timer = ::gridtools::ghex::reduce(timer, context.mpi_comm());
    if (comm.rank() == 0)
        std::cout << "all ranks: mean exchange time compact:                  " << global_timer.mean()/1000
                  << " ± " << global_timer.stddev()/1000 << " ms" << std::endl;
}

template<typename Context, typename Communicator, typename Pattern, typename Fields>
void run_sequence_Nco(Context& context, Communicator comm, Pattern& pattern, Fields& fields) {
    // communication object
    auto co_0 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_1 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_2 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_3 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_4 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_5 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_6 = gridtools::ghex::make_communication_object<Pattern>(comm);
    auto co_7 = gridtools::ghex::make_communication_object<Pattern>(comm);
    timer_type timer;
    // exchange
    co_0.pack_unpack(pattern(fields[0]));
    co_1.pack_unpack(pattern(fields[1]));
    co_2.pack_unpack(pattern(fields[2]));
    co_3.pack_unpack(pattern(fields[3]));
    co_4.pack_unpack(pattern(fields[4]));
    co_5.pack_unpack(pattern(fields[5]));
    co_6.pack_unpack(pattern(fields[6]));
    co_7.pack_unpack(pattern(fields[7]));
    for (int i=0; i<num_repetitions; ++i)
    {
        timer.tic();
        co_0.pack_unpack(pattern(fields[0]));
        co_1.pack_unpack(pattern(fields[1]));
        co_2.pack_unpack(pattern(fields[2]));
        co_3.pack_unpack(pattern(fields[3]));
        co_4.pack_unpack(pattern(fields[4]));
        co_5.pack_unpack(pattern(fields[5]));
        co_6.pack_unpack(pattern(fields[6]));
        co_7.pack_unpack(pattern(fields[7]));
        timer.toc();
    }
    if (comm.rank() == 0){
        std::cout << "rank 0:    mean exchange time sequenced (multiple CO):  " << timer.mean()/1000
                  << " ± " << timer.stddev()/1000 << " ms" << std::endl;
        std::cout << "rank 0: init  " << co_0.tinit.mean()/1000 << std::endl;
        std::cout << "rank 0: pack  " << co_0.tpack.mean()/1000 << std::endl;
        std::cout << "rank 0: upack " << co_0.tupack.mean()/1000 << std::endl;
    }
    auto global_timer = ::gridtools::ghex::reduce(timer, context.mpi_comm());
    if (comm.rank() == 0)
        std::cout << "all ranks: mean exchange time sequenced (multiple CO):  " << global_timer.mean()/1000
                  << " ± " << global_timer.stddev()/1000 << " ms" << std::endl;
}

template<typename Context, typename Communicator, typename Pattern, typename Fields>
void run_sequence_1co(Context& context, Communicator comm, Pattern& pattern, Fields& fields) {
    // communication object
    auto co = gridtools::ghex::make_communication_object<Pattern>(comm);
    timer_type timer;
    // exchange
    co.pack_unpack(pattern(fields[0]));
    co.pack_unpack(pattern(fields[1]));
    co.pack_unpack(pattern(fields[2]));
    co.pack_unpack(pattern(fields[3]));
    co.pack_unpack(pattern(fields[4]));
    co.pack_unpack(pattern(fields[5]));
    co.pack_unpack(pattern(fields[6]));
    co.pack_unpack(pattern(fields[7]));
    for (int i=0; i<num_repetitions; ++i)
    {
        timer.tic();
        co.pack_unpack(pattern(fields[0]));
        co.pack_unpack(pattern(fields[1]));
        co.pack_unpack(pattern(fields[2]));
        co.pack_unpack(pattern(fields[3]));
        co.pack_unpack(pattern(fields[4]));
        co.pack_unpack(pattern(fields[5]));
        co.pack_unpack(pattern(fields[6]));
        co.pack_unpack(pattern(fields[7]));
        timer.toc();
    }
    if (comm.rank() == 0){
        std::cout << "rank 0:    mean exchange time sequenced (single CO):    " << timer.mean()/1000
                  << " ± " << timer.stddev()/1000 << " ms" << std::endl;
        std::cout << "rank 0: init  " << co.tinit.mean()/1000 << std::endl;
        std::cout << "rank 0: pack  " << co.tpack.mean()/1000 << std::endl;
        std::cout << "rank 0: upack " << co.tupack.mean()/1000 << std::endl;
    }
    auto global_timer = ::gridtools::ghex::reduce(timer, context.mpi_comm());
    if (comm.rank() == 0)
        std::cout << "all ranks: mean exchange time sequenced (single CO):    " << global_timer.mean()/1000
                  << " ± " << global_timer.stddev()/1000 << " ms" << std::endl;
}

template<typename Context, typename Communicator, typename Pattern, typename Fields>
void run_rma(Context& context, Communicator comm, Pattern& pattern, Fields& fields) {
    // communication object
    auto co = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(
        comm,
        pattern(fields[0]),
        pattern(fields[1]),
        pattern(fields[2]),
        pattern(fields[3]),
        pattern(fields[4]),
        pattern(fields[5]),
        pattern(fields[6]),
        pattern(fields[7]));
    timer_type timer;
    // exchange
    co.pack_unpack();
    for (int i=0; i<num_repetitions; ++i)
    {
        timer.tic();
        co.pack_unpack();
        timer.toc();
    }
    if (comm.rank() == 0){
        std::cout << "rank 0:    mean exchange time RMA:                      " << timer.mean()/1000
                  << " ± " << timer.stddev()/1000 << " ms" << std::endl;
        std::cout << "rank 0: init  " << co.tinit.mean()/1000 << std::endl;
        std::cout << "rank 0: pack  " << co.tpack.mean()/1000 << std::endl;
        std::cout << "rank 0: upack " << co.tupack.mean()/1000 << std::endl;
    }
    auto global_timer = ::gridtools::ghex::reduce(timer, context.mpi_comm());
    if (comm.rank() == 0)
        std::cout << "all ranks: mean exchange time RMA:                      " << global_timer.mean()/1000
                  << " ± " << global_timer.stddev()/1000 << " ms" << std::endl;
}

template<typename Context, typename Communicator, typename Pattern, typename Fields>
void run_rma_sequenced(Context& context, Communicator comm, Pattern& pattern, Fields& fields) {
    // communication object
    auto co_0 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[0]));
    auto co_1 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[1]));
    auto co_2 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[2]));
    auto co_3 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[3]));
    auto co_4 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[4]));
    auto co_5 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[5]));
    auto co_6 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[6]));
    auto co_7 = gridtools::ghex::make_bulk_communication_object<context_type::bulk_exchange_type>(comm, pattern(fields[7]));
    timer_type timer;
    // exchange
    co_0.pack_unpack();
    co_1.pack_unpack();
    co_2.pack_unpack();
    co_3.pack_unpack();
    co_4.pack_unpack();
    co_5.pack_unpack();
    co_6.pack_unpack();
    co_7.pack_unpack();
    for (int i=0; i<num_repetitions; ++i)
    {
        timer.tic();
        co_0.pack_unpack();
        co_1.pack_unpack();
        co_2.pack_unpack();
        co_3.pack_unpack();
        co_4.pack_unpack();
        co_5.pack_unpack();
        co_6.pack_unpack();
        co_7.pack_unpack();
        //co_0.start_pack_unpack();
        //co_1.start_pack_unpack();
        //co_2.start_pack_unpack();
        //co_3.start_pack_unpack();
        //co_4.start_pack_unpack();
        //co_5.start_pack_unpack();
        //co_6.start_pack_unpack();
        //co_7.start_pack_unpack();
        //co_0.end_pack_unpack();
        //co_1.end_pack_unpack();
        //co_2.end_pack_unpack();
        //co_3.end_pack_unpack();
        //co_4.end_pack_unpack();
        //co_5.end_pack_unpack();
        //co_6.end_pack_unpack();
        //co_7.end_pack_unpack();
        timer.toc();
    }
    if (comm.rank() == 0){
        std::cout << "rank 0:    mean exchange time RMA sequenced:            " << timer.mean()/1000
                  << " ± " << timer.stddev()/1000 << " ms" << std::endl;
    }
    auto global_timer = ::gridtools::ghex::reduce(timer, context.mpi_comm());
    if (comm.rank() == 0)
        std::cout << "all ranks: mean exchange time RMA sequenced:            " << global_timer.mean()/1000
                  << " ± " << global_timer.stddev()/1000 << " ms" << std::endl;
}

TEST(CommunicationObjects, strategies) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int dims[3] = {0,0,0};
    int coords[3];
    int period[3] = {1,1,1};
    MPI_Comm CartComm;
    MPI_Dims_create(world_size, 3, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);
    MPI_Cart_get(CartComm, 3, dims, period, coords);
    std::array<bool, 3> periodic{ true, true, true };

    // make memory
    std::vector<float_type*> field_memory(num_fields);
    for (auto& ptr : field_memory)
        ptr = new float_type[num_grid_points];

    // compute global domain
    const std::array<int,3> global_domain = {local_dims[0]*dims[0], local_dims[1]*dims[1], local_dims[2]*dims[2]};
    const std::array<int,3> global_first = {0,0,0};
    const std::array<int,3> global_last = {global_domain[0]-1, global_domain[1]-1, global_domain[2]-1};
    // compute sub-domain coordinates in global frame
    const std::array<int,3> local_first = {local_dims[0]*coords[0], local_dims[1]*coords[1], local_dims[2]*coords[2]};
    const std::array<int,3> local_last = {local_dims[0]*(coords[0]+1)-1, local_dims[1]*(coords[1]+1)-1, local_dims[2]*(coords[2]+1)-1};
        
    {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, CartComm);
    auto& context = *context_ptr;
    auto comm = context.get_communicator(context.get_token());
    
    // define local domain
    domain_descriptor_type local_domain{comm.rank(), local_first, local_last};
    std::vector<domain_descriptor_type> local_domains{local_domain};

    // make pattern
    auto halo_gen = domain_descriptor_type::halo_generator_type(global_first, global_last, halo_vec, periodic, true);
    auto pattern = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    // wrap fields
    using field_type = decltype(gridtools::ghex::wrap_field<arch_type,2,1,0>(comm.rank(), (float_type*)0, offset, local_dims_extended));
    std::vector<field_type> fields;
    fields.reserve(num_fields);
    for (auto ptr : field_memory)
        fields.push_back(gridtools::ghex::wrap_field<arch_type,2,1,0>(comm.rank(), ptr, offset, local_dims_extended));

    run_compact(context, comm, pattern, fields);
    run_sequence_Nco(context, comm, pattern, fields);   
    run_sequence_1co(context, comm, pattern, fields);
    
    //run_rma(context, comm, pattern, fields);

    //run_rma_sequenced(context, comm, pattern, fields);

    }

    // free memory
    for (auto ptr : field_memory)
        delete[] ptr;

    MPI_Comm_free(&CartComm);
}
