!This module contains several module variables (see comments below)
!and two subroutines:
!simulate_jacobi: Uses jacobi iteration to compute solution
!to contamination model problem
!simulate: To be completed. Use over-step iteration method to
!simulate contamination model

module bmodel
    implicit none
    integer :: bm_kmax=10000 !max number of iterations
    real(kind=8) :: bm_tol=0.00000001d0 !convergence criterion
    real(kind=8), allocatable, dimension(:) :: deltaC !|max change in C| each iteration
    real(kind=8) :: bm_g=1.d0,bm_kbc=1.d0 !death rate, r=1 boundary parameter
    real(kind=8) :: bm_s0=1.d0,bm_r0=2.d0,bm_t0=1.5d0 !source parameters

contains
!-----------------------------------------------------
!Solve 2-d contaminant spread problem with Jacobi iteration
subroutine simulate_jacobi(n,C)
    !input  n: number of grid point (n+2 x n+2) grid
    !output C: final concentration field
    !       deltac(k): max(|C^k - C^k-1|)
    !A number of module variables can be set in advance.

    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    integer :: i1,j1,k1
    real(kind=8) :: pi,del,del2f
    real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm

    if (allocated(deltac)) then
      deallocate(deltac)
    end if
    allocate(deltac(bm_kmax))

    pi = acos(-1.d0)

    !grid--------------
    del = pi/dble(n+1)
    del2f = 0.25d0*(del**2)


    do i1=0,n+1
        r(i1,:) = 1.d0+i1*del
    end do

    do j1=0,n+1
        t(:,j1) = j1*del
    end do
    !-------------------

    !Update equation parameters------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2
    !-----------------

    !set initial condition/boundary conditions
    C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac
    !Jacobi iteration
    do k1=1,bm_kmax
        Cnew(1:n,1:n) = Sdel2(1:n,1:n) + C(2:n+1,1:n)*facp(1:n,1:n) + C(0:n-1,1:n)*facm(1:n,1:n) + &
                                         (C(1:n,0:n-1) + C(1:n,2:n+1))*fac2(1:n,1:n)
        deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
        C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
        if (deltac(k1)<bm_tol) exit !check convergence criterion
        if (mod(k1,1000)==0) print *, k1,deltac(k1)
    end do

    print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))

end subroutine simulate_jacobi
!-----------------------------------------------------

!Solve 2-d contaminant spread problem with over-step iteration method
subroutine simulate(n,C)
  !input  n: number of grid point (n+2 x n+2) grid
  !output C: final concentration field
  !       deltac(k): max(|C^k - C^k-1|)
  !A number of module variables can be set in advance.
  integer, intent(in) :: n
  real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
  integer :: i1,j1,k1,i2,i3
  real(kind=8) :: pi,del,del2f
  real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,fac,fac2,facp,facm
  real(kind=8), allocatable, dimension(:,:) :: Cnew
  integer(kind=8) :: clock_t1,clock_t2,clock_rate
  if (allocated(deltac)) then
    deallocate(deltac)
  end if
  allocate(deltac(bm_kmax))

  call system_clock(clock_t1)

  pi = acos(-1.d0)

  !grid--------------
  del = pi/dble(n+1)
  del2f = 0.25d0*(del**2)


  do i1=0,n+1
      r(i1,:) = 1.d0+i1*del
  end do

  do j1=0,n+1
      t(:,j1) = j1*del
  end do
  !-------------------

  !Update equation parameters------
  rinv2 = 1.d0/(r**2)
  fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
  facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
  facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
  fac2 = fac*rinv2
  !-----------------

  !set initial condition/boundary conditions
  C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
  Cnew = C
  !set source function, Sdel2 = S*del^2*fac
  Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac
  !Jacobi iteration
  do k1=1,bm_kmax
    do i2 = 1,n
      do i3 = 1,n
        Cnew(i2,i3) = 0.5d0*(-C(i2,i3)+3.d0*(Sdel2(i2,i3) + C(i2+1,i3)*facp(i2,i3) + fac2(i2,i3)*C(i2,i3+1)))
        Cnew(i2,i3) = Cnew(i2,i3) + 1.5d0*(facm(i2,i3)*Cnew(i2-1,i3)+fac2(i2,i3)*Cnew(i2,i3-1))
      end do
    end do
    deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
    C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
    if (deltac(k1)<bm_tol) exit !check convergence criterion
    if (mod(k1,1000)==0) print *, k1,deltac(k1)
  end do

  print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))

  call system_clock(clock_t2,clock_rate)
  print *, 'elapsed wall time (seconds)= ', dble(clock_t2-clock_t1)/dble(clock_rate)


end subroutine simulate

!----------------------------------------------------------------------------------------------------

subroutine theta_star(n,theta,C)
  !input  n: number of grid point (n+2 x n+2) grid
  !output C: final concentration field
  !       deltac(k): max(|C^k - C^k-1|)
  !A number of module variables can be set in advance.
  integer, intent(in) :: n
  real(kind=8), intent(in) :: theta
  real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
  integer :: i1,j1,k1,i2,i3
  real(kind=8) :: pi,del,del2f
  real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,fac,fac2,facp,facm
  real(kind=8), allocatable, dimension(:,:) :: Cnew
  integer(kind=8) :: clock_t1,clock_t2,clock_rate
  if (allocated(deltac)) then
    deallocate(deltac)
  end if
  allocate(deltac(bm_kmax))

  call system_clock(clock_t1)

  pi = acos(-1.d0)

  !grid--------------
  del = pi/dble(n+1)
  del2f = 0.25d0*(del**2)


  do i1=0,n+1
      r(i1,:) = 1.d0+i1*del
  end do

  do j1=0,n+1
      t(:,j1) = j1*del
  end do
  !-------------------

  !Update equation parameters------
  rinv2 = 1.d0/(r**2)
  fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
  facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
  facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
  fac2 = fac*rinv2
  !-----------------

  !set initial condition/boundary conditions
  C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
  C(0,0:n+1) = exp(-10.d0*(t(0,0:n+1)-theta)**2)*(sin(bm_kbc*t(0,0:n+1))**2)

  Cnew = C
  !set source function, Sdel2 = S*del^2*fac
  Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac
  !Jacobi iteration
  do k1=1,bm_kmax
    do i2 = 1,n
      do i3 = 1,n
        Cnew(i2,i3) = 0.5d0*(-C(i2,i3)+3.d0*(Sdel2(i2,i3) + C(i2+1,i3)*facp(i2,i3) + fac2(i2,i3)*C(i2,i3+1)))
        Cnew(i2,i3) = Cnew(i2,i3) + 1.5d0*(facm(i2,i3)*Cnew(i2-1,i3)+fac2(i2,i3)*Cnew(i2,i3-1))
      end do
    end do
    deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
    C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
    if (deltac(k1)<bm_tol) exit !check convergence criterion
    if (mod(k1,1000)==0) print *, k1,deltac(k1)
  end do

  print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))

  call system_clock(clock_t2,clock_rate)
  print *, 'elapsed wall time (seconds)= ', dble(clock_t2-clock_t1)/dble(clock_rate)


end subroutine theta_star


end module bmodel
