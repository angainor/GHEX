#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
#include <vector>
#include <iomanip>

#include <gtest/gtest.h>

int rank;

/**
 * Simple Send recv on two ranks. P0 sends a message, P1 receives it and check the content.
 */

void test1() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    if ( rank == 0 ) {
        //sr.blocking_send(smsg, 1, 1);
        sr.send(1, 1, smsg).get();
    } else if (rank == 1) {
        auto fut = sr.recv(0, 1, rmsg);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (!fut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int j = 1;
        for (auto i : rmsg) {
            EXPECT_EQ(static_cast<int>(i), j);
            ++j;
        }
    }
}

void test2() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(sr)>;
    gridtools::ghex::tl::callback_communicator<comm_type,allocator_type> cb_comm(sr);

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    smsg_type rmsg(10);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(1, 1, smsg);
        fut.wait();
    } else if (rank == 1) {
        cb_comm.recv(0, 1, rmsg, [ &arrived](int /*src*/, int /* tag */, const smsg_type&) { arrived = true; });

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            cb_comm.progress();
         } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int j = 1;
        for (auto i : rmsg) {
            EXPECT_EQ(static_cast<int>(i), j);
            ++j;
        }
    }

    EXPECT_FALSE(cb_comm.progress());

}

void test1_mesg() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;

    gridtools::ghex::tl::message_buffer<> smsg{40};

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::tl::message_buffer<> rmsg{40};

    if ( rank == 0 ) {
        sr.send(1, 1, smsg).get();
    } else if (rank == 1) {
        auto fut = sr.recv(0, 1, rmsg);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (!fut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int* data = rmsg.data<int>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }
}

void test2_mesg() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;
    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(sr)>;

    gridtools::ghex::tl::callback_communicator<comm_type,allocator_type> cb_comm(sr);

    gridtools::ghex::tl::message_buffer<> smsg{40};
    smsg_type rmsg{40};

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(1, 1, smsg);
        fut.wait();
    } else if (rank == 1) {
        cb_comm.recv(0, 1, rmsg, [ &arrived](int /* src */, int /* tag */, const smsg_type&) { arrived = true; });

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            cb_comm.progress();
         } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int* data = rmsg.data<int>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }

    EXPECT_FALSE(cb_comm.progress());

    MPI_Barrier(MPI_COMM_WORLD);

}

void test1_shared_mesg() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;

    gridtools::ghex::tl::message_buffer<> smsg{40};

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::tl::shared_message_buffer<> rmsg{40};

    if ( rank == 0 ) {
        sr.send(1, 1, smsg).get();
    } else if (rank == 1) {
        auto fut = sr.recv(0, 1, rmsg);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (!fut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int* data = rmsg.data<int>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }
}


template <typename Msg>
void print_msg(Msg const msg) {
    std::cout << "Reference count " << msg.use_count() << " (size: " << msg.size() << ")\n";
    int * data = msg.template data<int>();
    for (int i = 0; i < (int)(msg.size()/sizeof(int)); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << "\n";
}

TEST(transport, basic) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1();
}

TEST(transport, basic_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test2();
}

TEST(transport, basic_msg) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1_mesg();
}

TEST(transport, basic_msg_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test2_mesg();
}

TEST(transport, basic_shared_msg) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1_shared_mesg();
}

