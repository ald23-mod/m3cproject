Anas Lasri Doukkali
In this project we have added several new functions:

Part2:

Python
min_theta : (p2.py)This function calculates the optimal value of theta that optimizes
C_sum,  using scipy optimize

C_sum: (p2.py)This function calculates C_sum depending on theta, it is necessary for the above function

Fortran:
theta_star: This newly added routine to p2.f90 helps us calculate C_sum for different
values of theta, this is used as part of the analyze part of the code to be able to produce plots.
