PROGRAM test_send_recv_cb
  use iso_fortran_env
  use omp_lib
  use ghex_mod
  use ghex_comm_mod
  use ghex_message_mod
  use ghex_request_mod

  implicit none  

  include 'mpif.h'  

  integer :: mpi_err
  integer :: mpi_threading, mpi_size, mpi_rank, mpi_peer
  integer :: nthreads = 0, thrid

  type(ghex_communicator), dimension(:), pointer :: communicators
  type(ghex_communicator) :: comm

  ! message
  integer(8) :: msg_size = 16
  type(ghex_message) :: smsg, rmsg
  type(ghex_request) :: sreq
  integer(atomic_int_kind), allocatable :: received(:)[:]
  type(ghex_progress_status) :: ps
  integer(1), dimension(:), pointer :: msg_data

  procedure(f_callback), pointer :: pcb
  pcb => recv_callback

  call mpi_init_thread (MPI_THREAD_MULTIPLE, mpi_threading, mpi_err)
  call mpi_comm_size (mpi_comm_world, mpi_size, mpi_err)
  call mpi_comm_rank (mpi_comm_world, mpi_rank, mpi_err)
  if (mpi_size /= 2) then
    if (mpi_rank == 0) then
      print *, "Usage: this test can only be executed for 2 ranks"
    end if
    call mpi_finalize(mpi_err)
    call exit(0)
  end if
  mpi_peer = modulo(mpi_rank+1, 2)

  !$omp parallel shared(nthreads)
  nthreads = omp_get_num_threads()
  !$omp end parallel

  ! init ghex
  call ghex_init(nthreads, mpi_comm_world);

  ! initialize shared datastructures
  allocate(communicators(nthreads))
  allocate(received(nthreads)[*], source=0)

  ! make per-thread communicators
  !$omp parallel private(thrid, comm, sreq, smsg, rmsg, msg_data, ps)

  ! make thread id 1-based
  thrid = omp_get_thread_num()+1

  ! allocate a communicator per thread and store in a shared array
  communicators(thrid) = ghex_comm_new()
  comm = communicators(thrid)

  ! create messages, or get a reference to a shared message
  rmsg = ghex_message_new(msg_size, ALLOCATOR_STD)
  smsg = ghex_message_new(msg_size, ALLOCATOR_STD)

  ! initialize send data
  msg_data => ghex_message_data(smsg)
  msg_data(1:msg_size) = (mpi_rank+1)*10 + thrid;

  ! pre-post a recv. subsequent recv are posted inside the callback
  call ghex_comm_recv_cb(comm, rmsg, mpi_peer, thrid, pcb)

  ! send, but keep ownership of the message: buffer is not freed after send
  call ghex_comm_post_send_cb(comm, smsg, mpi_peer, thrid, req=sreq)

  ! progress the communication - complete the send before posting another one
  ! here we send the same buffer twice
  do while(.not.ghex_request_test(sreq))
    ps = ghex_comm_progress(comm)
  end do

  ! send again, give ownership of the message to ghex: buffer will be freed after completion
  call ghex_comm_send_cb(comm, smsg, mpi_peer, thrid, req=sreq)

  ! progress the communication - wait for all (2) recv to complete
  do while(minval(received)/=2)
    ps = ghex_comm_progress(comm)
  end do
  
  ! cleanup per-thread. messages are freed by ghex if comm_recv_cb and comm_send_cb
  call ghex_delete(communicators(thrid))

  !$omp end parallel

  ! cleanup shared
  deallocate(communicators)

  call ghex_finalize()  
  call mpi_finalize(mpi_err)

contains

  subroutine recv_callback (mesg, rank, tag) bind(c)
    use iso_c_binding
    type(ghex_message), value :: mesg
    type(ghex_request) :: req
    integer(c_int), value :: rank, tag
    integer :: thrid
    integer(1), dimension(:), pointer :: msg_data
    procedure(f_callback), pointer :: pcb
    pcb => recv_callback

    thrid = omp_get_thread_num()+1

    ! what have we received?
    msg_data => ghex_message_data(mesg)
    print *, mpi_rank, thrid, msg_data

    call atomic_add(received(tag), 1)

    ! resubmit if needed. here: receive only 2 (rank,tag) messages
    if (received(tag) < 2) then
      comm = communicators(thrid)
      call ghex_comm_resubmit_recv(comm, mesg, rank, tag, pcb)
      ! print *, "recv request ", rank, tag, " has been resubmitted"
    end if

  end subroutine recv_callback

END PROGRAM test_send_recv_cb