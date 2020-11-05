#include "context_bind.hpp"
#include <mpi.h>

context_uptr_type context;
int ghex_nthreads = 1;

extern "C"
void ghex_init(int nthreads, MPI_Fint fcomm)
{
    /* the fortran-side mpi communicator must be translated to C */
    MPI_Comm ccomm = MPI_Comm_f2c(fcomm);    
    context = ghex::tl::context_factory<transport>::create(ccomm);
    ghex_nthreads = nthreads;
}

extern "C"
void ghex_finalize()
{
    context.reset();
}

extern "C"
void ghex_obj_free(ghex::bindings::obj_wrapper **wrapper_ref)
{
    ghex::bindings::obj_wrapper *wrapper = *wrapper_ref;

    // clear the fortran-side variable
    *wrapper_ref = nullptr;
    delete wrapper;
}
