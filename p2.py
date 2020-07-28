"""Anas Lasri, CID:01209387
Final project, part 2"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
#from m1 import bmodel as bm #assumes p2.f90 has been compiled with: f2py -c p2.f90 -m m1


def simulate_jacobi(n,input_num=(10000,1e-8),input_mod=(1,1,1,2,1.5),display=True):
    """ Solve contamination model equations with
        jacobi iteration.
        Input:
            input_num: 2-element tuple containing kmax (max number of iterations
                        and tol (convergence test parameter)
            input_mod: 5-element tuple containing g,k_bc,s0,r0,t0 --
                        g: bacteria death rate
                        k_bc: r=1 boundary condition parameter
                        s0,r0,t0: source function parameters
            display: if True, a contour plot showing the final concetration field is generated
        Output:
            C,deltac: Final concentration field and |max change in C| each iteration
    """
    #Set model parameters------

    kmax,tol = input_num
    g,k_bc,s0,r0,t0 = input_mod
    #-------------------------------------------
    #Set Numerical parameters
    Del = np.pi/(n+1)
    r = np.linspace(1,1+np.pi,n+2)
    t = np.linspace(0,np.pi,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    #Factors used in update equation
    rinv2 = 1.0/(rg*rg)
    fac = 1.0/(2 + 2*rinv2+Del*Del*g)
    facp = (1+0.5*Del/rg)*fac
    facm = (1-0.5*Del/rg)*fac
    fac2 = fac*rinv2

    #set initial condition/boundary conditions
    C = (np.sin(k_bc*tg)**2)*(np.pi+1.-rg)/np.pi
    #set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*np.exp(-20.*((rg-r0)**2+(tg-t0)**2))*(Del**2)*fac

    deltac = []
    Cnew = C.copy()

    #Jacobi iteration
    for k in range(kmax):
        #Compute Cnew
        Cnew[1:-1,1:-1] = Sdel2[1:-1,1:-1] + C[2:,1:-1]*facp[1:-1,1:-1] + C[:-2,1:-1]*facm[1:-1,1:-1]+(C[1:-1,:-2] + C[1:-1,2:])*fac2[1:-1,1:-1] #Jacobi update
        #Compute delta_p
        deltac += [np.max(np.abs(C-Cnew))]
        C[1:-1,1:-1] = Cnew[1:-1,1:-1]
        if k%1000==0: print("k,dcmax:",k,deltac[k])
        #check for convergence
        if deltac[k]<tol:
            print("Converged,k=%d,dc_max=%28.16f " %(k,deltac[k]))
            break

    deltac = deltac[:k+1]


    if display:
        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Final concentration field')

    return C,deltac,k



def simulate(n,input_num=(10000,1e-8),input_mod=(1,1,1,2,1.5),display=True):
    """ Solve contamination model equations with
        OSI method, input/output same as in simulate_jacobi above
    """

    #Set model parameters------
    kmax,tol = input_num
    g,k_bc,s0,r0,t0 = input_mod

    #Set numerical parameters
    Del = np.pi/(n+1)
    r = np.linspace(1,1+np.pi,n+2)
    t = np.linspace(0,np.pi,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    #Factors used in update equation
    rinv2 = 1.0/(rg*rg)
    fac = 1.0/(2 + 2*rinv2+Del*Del*g)
    facp = (1+0.5*Del/rg)*fac
    facm = (1-0.5*Del/rg)*fac
    fac2 = fac*rinv2

    #set initial condition/boundary conditions
    C = (np.sin(k_bc*tg)**2)*(np.pi+1.-rg)/np.pi

    #set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*np.exp(-20.*((rg-r0)**2+(tg-t0)**2))*(Del**2)*fac
    deltac = []
    Cnew = C.copy()
    #OSI iteration
    for k in range(kmax):
        for i in range(1,n+1):
            for j in range(1,n+1):
                Cnew[i,j] = 0.5*(-C[i,j]+3.*(Sdel2[i,j] + C[i+1,j]*facp[i,j] + fac2[i,j]*C[i,j+1]))
                Cnew[i,j] = Cnew[i,j] + 1.5*(facm[i,j]*Cnew[i-1,j]+fac2[i,j]*Cnew[i,j-1])
        #Compute delta_p
        deltac += [np.max(np.abs(C-Cnew))]
        C[1:-1,1:-1] = Cnew[1:-1,1:-1]
        if k%1000==0: print("k,dcmax:",k,deltac[k])
        #check for convergence
        if deltac[k]<tol:
            print("Converged,k=%d,dc_max=%28.16f " %(k,deltac[k]))
            break

    deltac = deltac[:k+1]

    if display:
        plt.figure()
        plt.contour(t,r,C,50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Final concentration field')

    return C,deltac,k


def performance(n = [20,40,60,80,100,130]):
    """Analyze performance of simulation codes
    Add input/output variables as needed.
    """
    """We are asked to compare the working of four algorithms across two different
    languages(2 algorithms in Python, and two algorithms in Fortran), performance wise.
    We first know thus far that by principle the code implemented in the compiled
    languages(Fortran) will be innevitabely faster than the one implemented in the
    interpreted language(Fortran). This is as was explained in the beginning of the
    term because of the fact that when coding in compiled languages we are typing code
    that is already in the language of the machine.
    Moving on from this basic point. Our analysis of performance start by plotting a
    figue that compares all the 4 algorithms implemented on the basis of different values
    of n. Once this is done, we can clearly see that the least efficient code of the four
    as expected, is the Python OSI. This code where we use the OSI iteration method is
    the slowest as it was written in terms of loops and as seen before in the course. Loops
    tend to be really inneficient in Python. The second least efficient code, also comes
    as no surprise to be the Python Jacobi iteration method code. However we must note that
    the difference between the second least efficient code compared with both Fortran codes
    is much smaller than the difference between the OSI and Jacobi versions of the Python code.
    For the second plot I focus on the Fortran ALgorithms performance exclusively. This produces
    a plot that as expected shows that OSI is slower than its rival the Jacobi however by only a small
    difference that is orders or magnitude smaller than the difference between both Python codes.
    The third plot is just a repetition of the first plot however with only the python codes.
    This is done to emphasize the big difference in performance between both Python codes.
    I will now also compare as a performance feature the number of iterations it takes for the
    algorithms to output a solution. We have set this number to a maximum of kmax however in some
    cases where we acheive a good enough result(below the tolerance specified) then our algorithm
    is allowed to break before reaching kmax.

    """
    import m1
    #values for which the performance analysis will be done
    l = len(n)                       #to be used to define the dimension of variables


    #Initializing wall/run time vec. for different type of algorithms ------------------------
    wall_time_f90 = np.zeros(l,dtype=float)
    wall_time_f90j = np.zeros(l,dtype=float)
    wall_timepy = np.zeros(l,dtype=float)
    wall_timepy_jac = np.zeros(l,dtype=float)
    k_Jac = np.zeros(l)
    k_OSI = np.zeros(l)

    #Calculating run time for the Fortran OSI code--------------------------------------------
    for i in range(l):
        wall_time0 = time.time()
        m1.bmodel.simulate(n[i])
        wall_time_f90[i] = time.time() - wall_time0

    #Calculating run time for the Fortran Jacobi code-----------------------------------------
    for i in range(l) :
        wall_time0 = time.time()
        m1.bmodel.simulate_jacobi(n[i])
        wall_time_f90j[i] = time.time() - wall_time0

    #Calculating run time for the Python OSI code---------------------------------------------
    for i in range(l):
        wall_time0 = time.time()
        k_OSI[i] = simulate(n[i],display=False)[2]
        wall_timepy[i] = time.time() - wall_time0

    #Calculating run time for the Python OSI code---------------------------------------------
    for i in range(l):
        wall_time0 = time.time()
        k_Jac[i] = simulate_jacobi(n[i],display=False)[2]
        wall_timepy_jac[i] = time.time() - wall_time0

    #Code for plotting different figures------------------------------------------------------

    #The first plot will have a comparison of all running times for both the Python
    #and Fortran versions of the code.--------------------------------------------------------
    plt.hold(True)
    plt.plot(n,wall_time_f90,'x--')
    plt.plot(n,wall_time_f90j,'x--')
    plt.plot(n,wall_timepy_jac,'x--')
    plt.plot(n,wall_timepy,'x--')
    plt.title('Anas Lasri, CID:01209387 \n Time comparing 4 algorithm speeds')
    plt.xlabel('n (input of all functions for Bacterial contamination)')
    plt.ylabel('wall time')
    plt.legend(('fortran OSI', 'Fortran Jacobi','Python Jacobi','Python OSI'), loc = 'best')
    plt.savefig('part2_1',dpi=400)
    plt.show()

    #The second plot will have a comparison of only the Fortran subroutines------------------
    plt.plot(n,wall_time_f90j,'x--')
    plt.plot(n,wall_time_f90,'x--')
    plt.title('Anas Lasri, CID:01209387 \n Time comparing both Fortran subroutines')
    plt.xlabel('n (input of all functions for Bacterial contamination)')
    plt.ylabel('wall time')
    plt.legend(('Fortran Jacobian','Fortran OSI'), loc = 'best')
    plt.savefig('part2_2', dpi=400)
    plt.show()

    #The third plot will have a comparison of both the Python codes--------------------------
    plt.plot(n,wall_timepy_jac,'x--')
    plt.plot(n,wall_timepy,'x--')
    plt.title('Anas Lasri, CID:01209387 \n Time comparing both python subroutines' )
    plt.xlabel('n (input of all functions for Bacterial contamination)')
    plt.ylabel('wall time')
    plt.legend(('Python Jacobian','Python OSI'), loc = 'best')
    plt.savefig('part2_3', dpi=400)
    plt.show()

    #plot of the number of iterations
    plt.plot(n,k_Jac,'x--')
    plt.plot(n,k_OSI,'x--')
    plt.xlabel('N')
    plt.ylabel('Number of Iterations')
    plt.legend(('Jacobi iteration','OSI'), loc='best')
    plt.title('Anas Lasri, CID:01209387 \n NUmber of iterations plot')
    plt.savefig('part2_4',dpi=400)


    return None

def analyze(n,display=True):
    """Analyze influence of modified
    boundary condition on contamination dynamics
        Add input/output variables as needed.
    """
    """The idea is that I will try different theta's using the modified subroutine that I have
    created in fortran under the name theta_star. The main concern for this question is to
    choose the right value, if such value exists, that will minimize the contamination. I
    proceed to deal with this by first writing a function that will generate some plots for randomly
    selected values of theta_star. If by doing this a pattern is noticed then I will follow such
    pattern to lead me to the conclusion of which is the most appropriate theta_star. Otherwise,
    I will have a look at minimization methods, similar to the ones we have seen at the start of the
    module.
    From the first bit of analyzing the function by plotting a contour plot for different values of
    theta_star, we get insightful results however non deterministic. We see that the contour plots change
    drastically depending on the choice of this new parameter, theta_star. It is of high significance to
    note that adding a scale to the contour plot is important and this has been done in my code.

    p2_4_1:
    The first plot produced by running this functions with display=True is already showing you that there is a clear
    value for which theta_star minimized the contamination as you will also see from the scale of the
    contour plot where the highest value is approx 0.61

    I would like to clarify that another way to make the same analysis would be to include three dimensional
    plots instead of adding a scale for the contour plots. This as well is easily done however will be
    avoided as it is less efficient. Keeping in mind what has been done  first t oanalyze the question
    at hand, what I done next was to use scipy.minimize to be able to look for the optimal value
    of theta that minimizes the sum of the contamination levels. The function for these approach are
    defined underneath this same function. After obtaining said value of theta I proceeded to
    make a contour plot to be able to confirm and visualize and compare with the plots firstly produced.

    p2_4_2:
    The second plot plots the contour plots for the optimal value of theta as we get it from our
    minimize routine. As I mentioned earlier, we can plot this in a 3-d plot however we will not
    do so for efficiency reasons.

    p2_4_3:
    The last plot for this part of the project is one where I picked several values for k
    and plotted them against several values in the range of theta_star while using the fortran
    subroutine we created specifically for this purpose a while back.
    We can see in this last plot where we plotted C_sum against theta_star, symmetry which shows
    us something really important, there isn't a single minimizing value for theta_star as you can
    see, form both ends of the x axis, our C_sum value tends to be minimized.
    Furthermore, we can see that the oscilations increase as k increases in direct proportionality
    and also as k>>6 we can see that the curves smooths out for the most part and does not
    follow an ascillatory behaviour.
    """
    #Importing Fortran module
    import m1
    #Defining r and t for contour plot
    r = np.linspace(1,1+np.pi,n+2)
    t = np.linspace(0,np.pi,n+2)
    #Defining number of theta_star's we will try
    l = n
    C = np.zeros((n+2,n+2,l),dtype=float)
    #Defining the range of the theta_star's we will try
    theta_star = np.linspace(0.,np.pi,l)
    #Loop to plot contour depending on each theta_star from above
    for i in range(l):
        C[:,:,i] = m1.bmodel.theta_star(n,theta_star[i])

    fig = plt.figure()
    for i in range(1,5):
        if display:
            plt.hold(True)
            plt.subplot(2,2,i)
            plt.contour(t,r,C[:,:,i-1],n)
            #Adding the scale for the contour plot
            plt.colorbar()
            plt.xlabel('theta')
            plt.ylabel('r')
            plt.title('Final concentration field')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Anas Lasri, CID:01209387')
        plt.savefig('p2_4_1',dpi=400)
    #-----------------------------------------------------------------------------------------
    #From the figure that the code above outputs we can see that the change reaches optimal
    #minimal value for C when theta_star is near the half point(approx. 2.1). However now I will
    #try to figure out the exact value. To do so, I will create a new Python function.
    # THIS SHOWS THE OPTIMAL VALUE FOR THETA
    t_star = min_theta()
    C_star = m1.bmodel.theta_star(n,np.pi)
    fig = plt.figure()
    plt.contour(t,r,C_star,n)
    plt.colorbar()
    plt.xlabel('theta')
    plt.ylabel('r')
    plt.title('best theta_star for which C is minimized')
    plt.savefig('p2_4_2',dpi=400)
    plt.show()


    m1.bmodel.bm_g = 1
    m1.bmodel.bm_r0 = 1+np.pi/2
    m1.bmodel.bm_s0 = 2
    k = [1,2,3,4,5,6,7]
    Ck_sum = np.zeros((l,len(k)),dtype=float) #array of C sums for different k

    for i in range(len(k)):
        for j in range(l):
            m1.bmodel.bm_kbc = k[i]
            C = m1.bmodel.theta_star(n,theta_star[j])
            Ck_sum[j,i] = np.sum(C)

    plt.hold(True)
    for i in range(len(k)):
        plt.plot(theta_star,Ck_sum[:,i])
    plt.hold(False)
    plt.xlabel('\u03F4*')
    plt.ylabel('C sum ')
    plt.legend(('k=1','k=2','k=3','k=4','k=5','k=6','k=7'),loc='best')
    plt.savefig('p2_4_3',dpi=400)



    return None
#THis function defines the sum of the contamination
def C_sum(x):
    import m1
    C = m1.bmodel.theta_star(50,x)
    C_sum = np.sum(C)

    return C_sum
#This function minimizes the sum of the contamination defined above.
def min_theta():
    x0 = 2.
    res = minimize(C_sum,x0,method='nelder-mead',
                    options={'xtol': 1e-10, 'disp' : True})

    return res

if __name__=='__main__':
    #Add code below to call performance and analyze
    #and generate figures you are submitting in
    #your repo.
    input=()
