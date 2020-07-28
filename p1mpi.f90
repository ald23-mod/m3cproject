!Anas Lasri Doukkali, CID:01209387
!M3C 2018 Homework 4 part 1
!This module contains two module variables and four subroutines;
!two of these routines must be developed for this assignment.
!Module variables--
! nm_x: training images, typically n x d with n=784 and d<=4000
! nm_y: labels for training images, d-element array containing 0s and 1s
! corresponding to images of even and odd integers, respectively.
!
!Module routines---
! data_init: allocate nm_x and nm_y using input variables n and d. Used by sgd, may be used elsewhere if needed
! fgd_mpi: Use simple full gradient descent algorithm to iteratively find fitting parameters using nnmodel
! nnmodel_mpi: compute cost function and gradient using neural network model (NNM) with 1 output neuron and
! m neurons in the inner layer, and with nm_x and nm_y, and with fitting parameters provided as input

module nmodel_mpi
  use mpi
  implicit none
  real(kind=8), allocatable, dimension(:,:) :: nm_x
  integer, allocatable, dimension(:) :: nm_y

contains

!---allocate nm_x and nm_y deallocating first if needed (used by sgd)---
subroutine data_init(n,d)
  implicit none
  integer, intent(in) :: n,d
  if (allocated(nm_x)) deallocate(nm_x)
  if (allocated(nm_y)) deallocate(nm_y)
  allocate(nm_x(n,d),nm_y(d))
end subroutine data_init



!!Compute cost function and its gradient for neural network model
!for dlocal images (in nm_x) and dlocal labels (in nm_y) along with the
!fitting parameters provided as input in fvec. The network consists of
!an inner layer with m neurons and an output layer with a single neuron.
!fvec contains the elements of dw_inner, b_inner, w_outer, and b_outer
! Code has been provided below to "unpack" fvec
!Note: nm_x and nm_y must be allocated and set before calling this subroutine.
subroutine nnmodel_mpi(fvec,n,m,dlocal,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,m,dlocal,d !training data and inner layer sizes
  real(kind=8), dimension(m*(n+2)+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(m*(n+2)+1), intent(out) :: cgrad !gradient of cost
  integer :: i1,j1,l1
  real(kind=8), dimension(m,n) :: w_inner
  real(kind=8), dimension(m) :: b_inner,w_outer
  real(kind=8) :: b_outer
  !add variables as needed
  real(kind=8), dimension(m,dlocal) :: z_inner, a_inner, g_inner
  real(kind=8), dimension(dlocal) :: z_outer, a_outer, e_outer, g_outer,eg_outer
  real(kind=8) :: dcdb_outer
  real(kind=8),dimension(m) :: dcdw_outer
  real(kind=8), dimension(m) :: dcdb_inner
  real(kind=8), dimension(m,n) :: dcdw_inner
  real(kind=8) :: dinv
  dinv = 1.d0/dble(d)
  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*m+1
    w_inner(:,i1) = fvec(j1:j1+m-1) !inner layer weight matrix
  end do
  b_inner = fvec(n*m+1:n*m+m) !inner layer bias vector
  w_outer = fvec(n*m+m+1:n*m+2*m) !output layer weight vector
  b_outer  = fvec(n*m+2*m+1) !output layer bias
!----------------------------------------------------------------------
  !Compute portion of c and cgrad corresponding to dlocal images

  !Computer inner layer activation vector, a_inner
  z_inner  = matmul(w_inner,nm_x)
  do i1 = 1,dlocal
    z_inner(:,i1) = z_inner(:,i1) + b_inner
  end do
  a_inner = 1.d0/(1.d0 + exp(-z_inner))

  !Compute outer layer activation (a_outer) and cost
  z_outer = matmul(w_outer,a_inner) + b_outer
  a_outer = 1.d0/(1.d0 + exp(-z_outer))
  e_outer = a_outer - nm_y
  c = 0.5d0*dinv*sum((e_outer)**2)


  !Compute dc/db_outer and dc/dw_outer
  g_outer = a_outer*(1.d0 - a_outer)
  eg_outer = e_outer*g_outer
  dcdb_outer = dinv*sum(eg_outer)
  dcdw_outer = dinv*matmul(a_inner,eg_outer)

  !Compute dc/db_inner and dc/dw_inner
  g_inner = a_inner*(1.d0-a_inner)
  dcdb_inner = dinv*w_outer*matmul(g_inner,eg_outer)
  do l1 = 1,n
    dcdw_inner(:,l1) = dinv*w_outer*matmul(g_inner,(nm_x(l1,:)*eg_outer))
  end do

  !Pack gradient into cgrad
  do i1=1,n
    j1 = (i1-1)*m+1
    cgrad(j1:j1+m-1) = dcdw_inner(:,i1)
  end do
  cgrad(n*m+1:n*m+m) = dcdb_inner
  cgrad(n*m+m+1:n*m+2*m) = dcdw_outer
  cgrad(n*m+2*m+1) = dcdb_outer


end subroutine nnmodel_mpi

!Use full gradient descent
!to move towards optimal fitting parameters using
! nnmodel. Iterates for 400 "epochs" and final fitting
!parameters are stored in fvec.
!Input:
!fvec_guess: initial vector of fitting parameters
!n: number of pixels in each image (should be 784)
!m: number of neurons in inner layer; snmodel is used if m=0
!d: number of training images to be used; only the 1st d images and labels stored
!in nm_x and nm_y are used in the optimization calculation
!alpha: learning rate, it is fine to keep this as alpha=0.1 for this assignment
!Output:
!fvec: fitting parameters, see comments above for snmodel and nnmodel to see how
!weights and biases are stored in the array.
!Note: nm_x and nm_y must be allocated and set before calling this subroutine.

subroutine fgd_mpi(comm,fvec_guess,n,m,dlocal,d,alpha,fvec)
  implicit none
  integer, intent(in) :: comm,n,m,dlocal,d
  real(kind=8), dimension(:), intent(in) :: fvec_guess
  real(kind=8), intent(in) :: alpha
  real(kind=8), dimension(size(fvec_guess)), intent(out) :: fvec
  integer :: i1, j1, i1max=1000, n_fvec
  real(kind=8) :: c,c_total
  real(kind=8), dimension(size(fvec_guess)) :: cgrad, cgrad_total
  integer :: myid,ierr
  !Add other variables as needed

  integer :: numprocs
  call MPI_COMM_RANK(comm,myid,ierr)

  n_fvec = size(fvec_guess)
  fvec = fvec_guess
  do i1=1,i1max
    call nnmodel_mpi(fvec,n,m,dlocal,d,c,cgrad)
    call MPI_REDUCE( c,c_total,1,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
    !call MPI_REDUCE( cgrad, cgrad_total,n_fvec, MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
    !update fvec
    fvec = fvec - alpha*cgrad
    if ((myid==0) .and.(mod(i1,50)==0)) print *, 'completed epoch # ', i1, 'cost on process 0 =',c
  end do

end subroutine fgd_mpi



!Compute test error provided fitting parameters
!and with testing data stored in nm_x and nm_y
subroutine run_nnmodel(fvec,n,m,d,test_error)
  implicit none
  integer, intent(in) :: n,m,d !training data and inner layer sizes
  real(kind=8), dimension(m*(n+2)+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: test_error !test_error
  integer :: i1,j1
  real(kind=8), dimension(m,n) :: w_inner
  real(kind=8), dimension(m) :: b_inner,w_outer
  real(kind=8) :: b_outer
  !Declare other variables as needed
  real(kind=8), dimension(m,d) :: z_inner,a_inner
  real(kind=8), dimension(d) :: z_outer,a_outer
  integer, dimension(d) :: e_outer


  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*m+1
    w_inner(:,i1) = fvec(j1:j1+m-1) !inner layer weight matrix
  end do
  b_inner = fvec(n*m+1:n*m+m) !inner layer bias vector
  w_outer = fvec(n*m+m+1:n*m+2*m) !output layer weight vector
  b_outer  = fvec(n*m+2*m+1) !output layer bias

  !Add code to compute c and cgrad

  !Compute inner layer activation vector, a_inner
  z_inner = matmul(w_inner,nm_x)
  do i1=1,d
    z_inner(:,i1) = z_inner(:,i1) + b_inner
  end do
  a_inner = 1.d0/(1.d0 + exp(-z_inner))

  !Compute outer layer activation (a_outer) and cost
  z_outer = matmul(w_outer,a_inner) + b_outer
  a_outer = 1.d0/(1.d0+exp(-z_outer))
  e_outer = nint(abs(a_outer-nm_y))

  test_error = dble(sum(e_outer))/dble(d)


end subroutine run_nnmodel


end module nmodel_mpi
