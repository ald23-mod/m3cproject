!Anas Lasri Doukkali, CID:01209387
! This is a main program which should be used with the nmodel_mpi module
!
! It reads the first 5000 images and corresponding labels from data.csv
! into the variable, p, which is then split into the
! images (xfull) and labels (yfull)
! The subroutine fgd_mpi is called with the first dtrain images+labels;
! which will the use nnmodel to refine the initially random fitting
! parameters.
! Finally, the fitting parameters
! returned by fgd are written to the file, fvec.txt
! To compile: mpif90 -o p1mpi.exe p1mpi.f90 p1main_mpi.f90
! To run : mpiexec -n 4 p1mpi.exe

program p1main_mpi
  use mpi
  use nmodel_mpi !brings in module variables x and y as well as the module subroutines
  implicit none
  integer, parameter :: n=784,d=5000, dtest=1000
  integer :: m=4
  integer :: i1,j1,dtrain=5000, p(n+1,d)
  real(kind=8) :: xfull(n,d),xtest(n,dtest), test_error
  real(kind=8), allocatable, dimension(:) :: fvec0,fvec !array of fitting parameters
  integer :: yfull(d),ytest(dtest) !Labels
  integer :: myid, numprocs, ierr
  integer :: istart,iend,dlocal
  integer :: s_fvec

  ! Initialize MPI
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

  if (myid==0) print *, 'numprocs=',numprocs

  !read raw data from data.csv and store in p
  open(unit=12,file='data.csv')
  do i1=1,d
    read(12,*) p(:,i1)
    if ((myid==0) .and. (mod(i1,1000)==0)) print *, 'read in image #', i1
  end do
  close(12)

  open(unit=10,file='data.in')
    read(10,*) dtrain !must have d >= dtrain + dtest
    read(10,*) m
  close(10)

  call MPE_DECOMP1D( dtrain, numprocs, myid, istart, iend)
  dlocal = iend-istart+1 !Each process should sum over dlocal images when computing cgrad


  !Rearrange into input data, x,  and labels, y
  xfull = p(1:n,:)/255.d0
  yfull = p(n+1,:)
  yfull = mod(yfull,2)
  print *, 'yfull(1:4)',yfull(1:4) ! output first few labels
  xtest = xfull(:,dlocal-dtest+1:dlocal) ! set up test data (not used below)
  ytest = yfull(dlocal-dtest+1:dlocal)

  !NNM, d training images---------------
  call data_init(n,dlocal) !allocate module variables x and y
  nm_x = xfull(:,istart:iend) !set module variables
  nm_y = yfull(istart:iend)
  allocate(fvec0(m*(n+2)+1),fvec(m*(n+2)+1))


  !generate initial guess, code must
  !be added to provide copies of this guess
  !to other processes
  if (myid==0) then
    call random_number(fvec0) !set initial fitting parameters
    fvec0 = fvec0-0.5d0
  end if
  s_fvec = size(fvec0)
  !This is then key part for the first half of  part 1
  !We use the Broadcast function in MPI to send a copy of the initial guess to
  !all processes, This follows basic syntax for the MPI_BCAST routine
  call MPI_BCAST( fvec0,s_fvec,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

  !Use stochastic gradient descent, setting m>0 will use nnmodel instead of snmodel within sgd subroutine
  call fgd_mpi(MPI_COMM_WORLD,fvec0,n,m,dlocal,dtrain,2.5d0,fvec) !requires snmodel subroutine to be operational

  if (myid==0) then
    !write fitting parameters to file, fvec.txt
    !Can be loaded into python with f = np.loadtxt('fvec.txt')
    open(unit=22,file='fvec.txt')
    do i1=1,m*(n+2)+1
      write(22,*) fvec(i1)
    end do
    close(22)

    !Compute test error
    call data_init(n,dtest)
    nm_x = xtest
    nm_y = ytest
    call run_nnmodel(fvec,n,m,dtest,test_error)
    print *, 'test_error=',test_error
  end if

  deallocate(nm_x,nm_y)
  call mpi_finalize(ierr)
end program p1main_mpi

!--------------------------------------------------------------------
!  (C) 2001 by Argonne National Laboratory.
!      See COPYRIGHT in online MPE documentation.
!  This file contains a routine for producing a decomposition of a 1-d array
!  when given a number of processors.  It may be used in "direct" product
!  decomposition.  The values returned assume a "global" domain in [1:n]
!
subroutine MPE_DECOMP1D( n, numprocs, myid, s, e )
    implicit none
    integer :: n, numprocs, myid, s, e
    integer :: nlocal
    integer :: deficit

    nlocal  = n / numprocs
    s       = myid * nlocal + 1
    deficit = mod(n,numprocs)
    s       = s + min(myid,deficit)
    if (myid .lt. deficit) then
        nlocal = nlocal + 1
    endif
    e = s + nlocal - 1
    if (e .gt. n .or. myid .eq. numprocs-1) e = n

end subroutine MPE_DECOMP1D
