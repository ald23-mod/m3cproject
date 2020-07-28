! This is a main program which can be used with the nmodel module
! if and as you wish as you develop your code.
! It reads the first 5000 images and corresponding labels from data.csv
! into the variable, p, which is then split into the
! images (xfull) and labels (yfull)
! The subroutine fgd is called with the first dtrain images+labels;
! which will the use nnmodel to refine the initially random fitting
! parameters.
! Finally, the fitting parameters
! returned by fgd are written to the file, fvec.txt
! To compile: gfortran -o p1.exe p1serial.f90 p1main.f90

program p1main
  use nmodel !brings in module variables x and y as well as the module subroutines
  implicit none
  integer, parameter :: n=784,d=5000, dtest=1000
  integer:: m
  integer :: i1, dtrain=4000, p(n+1,d)
  real(kind=8) :: xfull(n,d),xtest(n,dtest), test_error
  real(kind=8), allocatable, dimension(:) :: fvec0,fvec !array of fitting parameters
  integer :: yfull(d),ytest(dtest) !Labels


  !read raw data from data.csv and store in p
  open(unit=12,file='data.csv')
  do i1=1,d
    read(12,*) p(:,i1)
    if (mod(i1,1000)==0) print *, 'read in image #', i1
  end do
  close(12)

  open(unit=10,file='data.in')
    read(10,*) dtrain !must have d >= dtrain + dtest
    read(10,*) m
  close(10)


  !Rearrange into input data, x,  and labels, y
  xfull = p(1:n,:)/255.d0
  yfull = p(n+1,:)
  yfull = mod(yfull,2)
  print *, 'yfull(1:4)',yfull(1:4) ! output first few labels
  xtest = xfull(:,d-dtest+1:d) ! set up test data (not used below)
  ytest = yfull(d-dtest+1:d)

  !NNM, d training images---------------
  call data_init(n,dtrain) !allocate module variables x and y
  nm_x = xfull(:,1:dtrain) !set module variables
  nm_y = yfull(1:dtrain)
  allocate(fvec0(m*(n+2)+1),fvec(m*(n+2)+1))
  call random_number(fvec0) !set initial fitting parameters
  fvec0 = fvec0-0.5d0

  !Use stochastic gradient descent, setting m>0 will use nnmodel instead of snmodel within sgd subroutine
  call fgd(fvec0,n,m,dtrain,2.5d0,fvec) !requires snmodel subroutine to be operational

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

  deallocate(nm_x,nm_y)
end program p1main
