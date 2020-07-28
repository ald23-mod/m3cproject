!This module contains two module variables and three subroutines;
!None of these routines need to be modified for this assignment.
!Module variables--
! nm_x: training images, typically n x d with n=784 and d<=4000
! nm_y: labels for training images, d-element array containing 0s and 1s
! corresponding to images of even and odd integers, respectively.
!
!Module routines---
! data_init: allocate nm_x and nm_y using input variables n and d. Used by sgd, may be used elsewhere if needed
! fgd: Use simple full gradient descent algorithm to iteratively find fitting parameters using nnmodel
! nnmodel: compute cost function and gradient using neural network model (NNM) with 1 output neuron and
! m neurons in the inner layer, and with nm_x and nm_y, and with fitting parameters provided as input

module nmodel
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
!for d images (in nm_x) and d labels (in nm_y) along with the
!fitting parameters provided as input in fvec. The network consists of
!an inner layer with m neurons and an output layer with a single neuron.
!fvec contains the elements of dw_inner, b_inner, w_outer, and b_outer
subroutine nnmodel(fvec,n,m,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,m,d !training data and inner layer sizes
  real(kind=8), dimension(m*(n+2)+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(m*(n+2)+1), intent(out) :: cgrad !gradient of cost
  integer :: i1,j1,l1
  real(kind=8), dimension(m,n) :: w_inner
  real(kind=8), dimension(m) :: b_inner,w_outer
  real(kind=8) :: dinv,b_outer
  !Declare other variables as needed
  real(kind=8), dimension(m,d) :: z_inner,a_inner,g_inner
  real(kind=8), dimension(d) :: z_outer,a_outer,e_outer,g_outer,eg_outer
  real(kind=8) :: dcdb_outer
  real(kind=8),dimension(m) :: dcdw_outer
  real(kind=8), dimension(m) :: dcdb_inner
  real(kind=8), dimension(m,n) :: dcdw_inner

  dinv = 1.d0/dble(d)

  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*m+1
    w_inner(:,i1) = fvec(j1:j1+m-1) !inner layer weight matrix
  end do
  b_inner = fvec(n*m+1:n*m+m) !inner layer bias vector
  w_outer = fvec(n*m+m+1:n*m+2*m) !output layer weight vector
  b_outer  = fvec(n*m+2*m+1) !output layer bias


  !Compute inner layer activation vector, a_inner
  z_inner = matmul(w_inner,nm_x)
  do i1=1,d
    z_inner(:,i1) = z_inner(:,i1) + b_inner
  end do
  a_inner = 1.d0/(1.d0 + exp(-z_inner))

  !Compute outer layer activation (a_outer) and cost
  z_outer = matmul(w_outer,a_inner) + b_outer
  a_outer = 1.d0/(1.d0+exp(-z_outer))
  e_outer = a_outer-nm_y
  c = 0.5d0*dinv*sum((e_outer)**2)

  !Compute dc/db_outer and dc/dw_outer
  g_outer = a_outer*(1.d0-a_outer)
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

end subroutine nnmodel


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
subroutine fgd(fvec_guess,n,m,d,alpha,fvec)
  implicit none
  integer, intent(in) :: n,m,d
  real(kind=8), dimension(:), intent(in) :: fvec_guess
  real(kind=8), intent(in) :: alpha
  real(kind=8), dimension(size(fvec_guess)), intent(out) :: fvec
  integer :: i1, j1, i1max=1000
  real(kind=8) :: c
  real(kind=8), dimension(size(fvec_guess)) :: cgrad
  real(kind=8), dimension(d) :: a
  real(kind=8), dimension(d+1) :: r
  integer, dimension(d+1) :: j1array

  fvec = fvec_guess
  do i1=1,i1max
    call nnmodel(fvec,n,m,d,c,cgrad)
    fvec = fvec - alpha*cgrad !update fitting parameters using gradient descent step

    if (mod(i1,50)==0) print *, 'completed epoch # ', i1, 'cost=',c

  end do

end subroutine fgd



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

end module nmodel
