!-------------------------------
!Fortran program for simulating contamination
!model using AOS method combined with a
!distributed memory approach (with MPI)
!The main program 1) Initializes MPI , 2) Reads in
!numerical and model parameters from data.in (which must be created),
!3) calls simulate_mpi (which must be completed) and 4) writes
! the final concentration field to a file that can be loaded witn numpy
!
! simulate_mpi: Sets up domain decomposition allocating different radial segments
! of the domain to different processes. Then each process uses AOS iteration
! to solve the model communicating only as necessary.
! After completing iterations, the routine gathers the radial portions of
! the final concentration fields from each process onto process 0 and this
! full field is returned to the main program

!----------------
program part3_mpi
    use mpi
    implicit none

    integer :: n! number of grid points (corresponds to (n+2 x n+2) grid)
    integer :: kmax !max number of iterations
    real(kind=8) :: tol, g !convergence tolerance, death rate
    real(kind=8), allocatable, dimension(:) :: deltac
    real(kind=8), allocatable, dimension(:,:) :: C !concentration matrix
    real(kind=8) :: S0,r0,t0 !source parameters
    real(kind=8) :: k_bc !r=1 boundary condition parameter
    integer :: i1,j1
    integer :: myid, numprocs, ierr

 ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

!gather input
    open(unit=10,file='data.in')
        read(10,*) n
        read(10,*) kmax
        read(10,*) tol
        read(10,*) g
        read(10,*) S0
        read(10,*) r0
        read(10,*) t0
        read(10,*) k_bc
   close(10)

    allocate(C(0:n+1,0:n+1),deltaC(kmax))

!compute solution
    call simulate_mpi(MPI_COMM_WORLD,numprocs,n,kmax,tol,g,S0,r0,t0,k_bc,C,deltaC)
!output solution (after completion of gather in euler_mpi)
     call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      if (myid==0) then
        open(unit=11,file='f.dat')
        do i1=0,n+1
            write(11,('(1000E28.16)')) C(i1,:)
        end do
        close(11)

        open(unit=12,file='deltac.dat')
        do i1=1,kmax
	       write(12,*) deltac(i1)
	      end do
        close(12)
      end if
    !can be loaded in python with: c=np.loadtxt('C.dat')

    call MPI_FINALIZE(ierr)
end program part3_mpi


!Simulate contamination model with AOS iteration
subroutine simulate_mpi(comm,numprocs,n,kmax,tol,g,S0,r0,t0,k_bc,C,deltaC)
    !input:
    !comm: MPI communicator
    !numprocs: total number of processes
    !n: total number of grid points in each direction (actually n+2 points in each direction)
    !tol: convergence criteria
    !kmax: total number of iterations
    !g: bacteria death rate
    !S0,r0,t0: source function parameters
    !k_bc: r=1 boundary condition parameter
    !output: C, final solution
    !deltaC: |max change in C| each iteration
    use mpi
    implicit none
    integer, intent (in) :: comm,numprocs,n,kmax
    real(kind=8), intent(in) ::tol,g,S0,r0,t0,k_bc
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    real(kind=8), intent(out) :: deltac(kmax)
    integer :: i1,i2,j0,j1,k,istart,iend,i3,j2,j3,i4,j4,j5
    integer :: myid,ierr,nlocal,receiver,sender,request
    real(kind=8) :: t1d(0:n+1),del,pi
    real(kind=8), allocatable, dimension(:) :: r1d
    real(kind=8),allocatable, dimension(:,:) :: r,t,Clocal,Sdel2
    integer, dimension(numprocs) :: disps,Nper_proc
    real(kind=8), dimension(0:n+1,0:n+1) :: rinv2,fac,facp,facm,fac2,Cnew2,Cnew1
    integer, dimension(MPI_STATUS_SIZE) :: status

    call MPI_COMM_RANK(comm, myid, ierr)
    print *, 'start simulate_mpi, myid=',myid

    !Set up theta
    pi = acos(-1.d0)
    del = pi/dble(n+1)
    do i1 = 0,n+1
      t1d(i1) = i1*del
    end do


    !generate decomposition and allocate sub-domain variables
    !Note: this decomposition is for indices 1:n; it
    !ignores the boundaries at i=0 and i=n+1 which
    !should be assigned to the first and last processes
    call mpe_decomp1d(n,numprocs,myid,istart,iend)
    print *, 'istart,iend,threadID=',istart,iend,myid
    nlocal = iend-istart+1
    allocate(r1d(nlocal+2))

    if (myid<numprocs-1) then
      receiver = myid + 1
    else
      receiver = 0
    end if

    if (myid>0) then
      sender = myid - 1
    else
      sender = numprocs - 1
    end if

    !generate local grid and allocate Clocal----
    do i1=istart-1,iend+1
      r1d(i1-istart+2) = 1.d0 + i1*del
    end do

    allocate(r(0:nlocal+1,0:n+1),t(0:nlocal+1,0:n+1),Clocal(0:nlocal+1,0:n+1))
    do j1=0,n+1
      r(:,j1) = r1d
    end do
    do i1=0,nlocal+1
      t(i1,:) = t1d
    end do

    !  ----------------------------------------------------------
    !Update equation parameters----------------------------------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*g) !1/(del^2*k^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2

    !Set initial condition/boundary conditions-------------------
    Clocal = (sin(k_bc*t)**2)*(pi+1.d0-r)/pi
    Cnew1 = Clocal

    call MPI_ISEND(Clocal(nlocal+1,:), n+1, MPI_DOUBLE_PRECISION, receiver, 0, MPI_COMM_WORLD,request,ierr)
    call MPI_RECV(Clocal(1,:),n+1,MPI_DOUBLE_PRECISION,sender,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)


    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
    !Set source function, Sdel2 = S*del^2*fac--------------------
    Sdel2 = s0*exp(-20.d0*((r-r0)**2+(t-t0)**2))*(del**2)*fac

    !AOS iteration---
    do k = 1,kmax
      !White update:

      !both columns are rows to be odd
      Cnew1(1:n:2,1:n:2) = (Sdel2(1:n:2,1:n:2) + Clocal(2:n+1:2,1:n:2)*facp(1:n:2,1:n:2)                      + &
                         Clocal(1:n:2,2:n+1:2)*fac2(1:n:2,1:n:2) + Clocal(0:n-1:2,1:n:2)*facm(1:n:2,1:n:2) + &
                         Clocal(1:n:2,0:n-1:2)*fac2(1:n:2,1:n:2))*1.5d0 - Clocal(1:n:2,1:n:2)/2.d0

      !Both columns and rows to be even
      Cnew1(2:n:2,2:n:2) = (Sdel2(2:n:2,2:n:2) + Clocal(3:n+1:2,2:n:2)*facp(2:n:2,2:n:2)                      + &
                         Clocal(2:n:2,3:n+1:2)*fac2(2:n:2,2:n:2) + Clocal(1:n-1:2,2:n:2)*facm(2:n:2,2:n:2) + &
                         Clocal(2:n:2,1:n-1:2)*fac2(2:n:2,2:n:2))*1.5d0 - Clocal(2:n:2,2:n:2)/2.d0

      !BLack grid update :
      Cnew2 = Cnew1

      !This is for rows odd however even columns
      Cnew2(1:n:2,2:n:2) = (Cnew2(2:n+1:2,2:n:2)*facp(1:n:2,2:n:2) + Cnew2(1:n:2,3:n+1:2)*fac2(1:n:2,2:n:2) + &
                         Cnew2(0:n-1:2,2:n:2)*facm(1:n:2,2:n:2) + Cnew2(1:n:2,1:n-1:2)*fac2(1:n:2,2:n:2) + &
                         Sdel2(1:n:2,2:n:2))*1.5d0 - Cnew2(1:n:2,2:n:2)/2.d0

      !This is for even rows however odd columns
      Cnew2(2:n:2,1:n:2) = (Cnew2(3:n+1:2,1:n:2)*facp(2:n:2,1:n:2) + Cnew2(2:n:2,2:n+1:2)*fac2(2:n:2,1:n:2) + &
                         Cnew2(1:n-1:2,1:n:2)*facm(2:n:2,1:n:2) + Cnew2(2:n:2,0:n-1:2)*fac2(2:n:2,1:n:2) + &
                         Sdel2(2:n:2,1:n:2))*1.5d0 - Cnew2(2:n:2,1:n:2)/2.d0
      deltac(k) = maxval(abs(Cnew2-Clocal))
      Clocal = Cnew2
      if (deltac(k)<tol) exit
      if (mod(k,1000)==0) print *, k,deltac(k)
    end do
    print *, 'k,error=',k,deltaC(min(k,kmax))
    !---------------


    !---------------------------------------------------------
    !Code below constructs C from the Clocal on each process
    print *, 'before collection',myid, maxval(abs(Clocal))

    i1=1
    i2 = nlocal

    if (myid==0) then
      i1=0
      nlocal = nlocal+1
    elseif (myid==numprocs-1) then
      i2 = nlocal+1
      nlocal = nlocal + 1
    end if

    call MPI_GATHER(nlocal,1,MPI_INT,NPer_proc,1,MPI_INT,0,comm,ierr)
    !collect Clocal from each processor onto myid=0

    if (myid==0) then
        disps(1)=0
        do j1=2,numprocs
          disps(j1) = disps(j1-1) + Nper_proc(j1-1)*(n+2)
        end do

        print *, 'nper_proc=',NPer_proc
        print *, 'disps=',disps
    end if

  !collect Clocal from each processor onto myid=0

     call MPI_GATHERV(transpose(Clocal(i1:i2,:)),nlocal*(n+2),MPI_DOUBLE_PRECISION,C,Nper_proc*(n+2), &
                 disps,MPI_DOUBLE_PRECISION,0,comm,ierr)

      C = transpose(C)
    if (myid==0) print *, 'finished',maxval(abs(C)),sum(C)


end subroutine simulate_mpi



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
