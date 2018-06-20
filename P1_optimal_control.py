import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt

def q1_ode_fun(tau, z):
    # Return array containing RHS of ODEs
    # z = [x1 x2 x3 p1 p2 p3 r]
    # x1 = x
    # x2 = y
    # x3 = theta

    w = (-0.5)*z[5] # -p3/2
    V = (-0.5)*(z[3]*math.cos(z[2]) + z[4]*math.sin(z[2]))
    x_dot = z[6]*np.array([V*math.cos(z[2]), V*math.sin(z[2]), w])
    p_dot = z[6]*np.array([0, 0, z[3]*V*math.sin(z[2]) - z[4]*V*math.cos(z[2])]);
    r_dot = 0 # dummy state

    return np.hstack((x_dot,p_dot,r_dot))

def q1_bc_fun(za, zb):
    # Return a tuple with 2 arrays - left and right BCs
    # len(BC_left) + len(BC_right) = num_ODEs

    # lambda
    lambda_test = 0.249

    # goal pose
    x_g = 5
    y_g = 5
    th_g = -np.pi/2.0
    xf = [x_g, y_g, th_g]

    # initial pose
    x0 = [0, 0, -np.pi/2.0]

    # Code boundary condition residuals
    BC_left = np.array([za[0]-x0[0], za[1]-x0[1], za[2]-x0[2]])

    x = zb[0];
    y = zb[1];
    th = zb[2];
    p1 = zb[3];
    p2 = zb[4];
    p3 = zb[5];
    tf = zb[6];

    w = (-0.5)*p3;
    V = (-0.5)*(p1*math.cos(th) + p2*math.sin(th));
    H_f = (lambda_test + V**2 + w**2 + p1*V*math.cos(th) + p2*V*math.sin(th) + p3*w);

    BC_right = np.array([zb[0]-xf[0], zb[1]-xf[1], zb[2]-xf[2], H_f])

    return (BC_left, BC_right)

#Define solver state: z = [x, y, th, p1, p2, p3, r]
guess = (1.0, 1.0, -np.pi/4.0, 1.0, 0.0, 0.0, 5.0)

problem = scikits.bvp_solver.ProblemDefinition(num_ODE=7, #Number of ODes
                                               num_parameters = 0, #Number of parameters
                                               num_left_boundary_conditions = 3, #Number of left BCs
                                               boundary_points = (0,1), #Boundary points of independent coordinate
                                               function = q1_ode_fun, #ODE function
                                               boundary_conditions = q1_bc_fun) #BC function

soln = scikits.bvp_solver.solve(problem, solution_guess = guess, trace = 0)

dt = 0.005

# Test if time is reversed in bvp_solver solution
z_0 = soln(0)
flip = 0
if z_0[-1] < 0:
    t_f = -z_0[-1]
    flip = 1
else:
    t_f = z_0[-1]

t = np.arange(0,t_f,dt)
z = soln(t/t_f)
if flip:
    z[3:7,:] = -z[3:7,:]
z = z.T # solution arranged column-wise

# Recover optimal control histories
V = -0.5*(z[:,3]*np.array([math.cos(x) for x in z[:,2]]) +
          z[:,4]*np.array([math.sin(x) for x in z[:,2]]))
om = -0.5*z[:,5]

V = np.array([V]).T # Convert to 1D column matrices
om = np.array([om]).T

# Ensure solution feasible
assert max(abs(V)) <= 0.5
assert max(abs(om)) <= 1

# Save trajectory data (state and controls)
data = np.hstack((z[:,:3],V,om))
np.save('traj_data_optimal_control',data)

# Plots
plt.rc('font', weight='bold', size=16)

plt.figure()
plt.plot(z[:,0], z[:,1],'k-',linewidth=2)
plt.quiver(z[1:-1:200,0],z[1:-1:200,1],np.cos(z[1:-1:200,2]),np.sin(z[1:-1:200,2]))
plt.grid('on')
plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
plt.xlabel('X'); plt.ylabel('Y')

plt.figure()
plt.plot(t, V,linewidth=2)
plt.plot(t, om,linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'])

plt.show()
