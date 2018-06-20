import numpy as np
from numpy import linalg
from P3_pose_stabilization import ctrl_pose
import math

def ctrl_traj(x,y,th,ctrl_prev,x_d,y_d,xd_d,yd_d,xdd_d,ydd_d,x_g,y_g,th_g):
    # (x,y,th): current state
    # ctrl_prev: previous control input (V,om)
    # (x_d, y_d): desired position
    # (xd_d, yd_d): desired velocity
    # (xdd_d, ydd_d): desired acceleration
    # (x_g,y_g,th_g): desired final state

    # Timestep
    dt = 0.005

    # Gains
    kpx = 0.9
    kpy = 0.9
    kdx = 0.7
    kdy = 0.7

    # Define control inputs (V,om) - without saturation constraints
    # Switch to pose controller once "close" enough, i.e., when
    # the robot is within 0.5m of the goal xy position.
    if (math.sqrt((x-x_d)**2 + (y-y_d)**2) > 0.0):
        # Use virtual control law
        V_prev = ctrl_prev[0];
        om_prev = ctrl_prev[1];
        xd = V_prev*math.cos(th);
        yd = V_prev*math.sin(th);
        u1 = xdd_d + kpx*(x_d - x) + kdx*(xd_d - xd)
        u2 = ydd_d + kpy*(y_d - y) + kdy*(yd_d - yd)
    
        Vd = u1*math.cos(th) + u2*math.sin(th)
        V = V_prev + Vd*dt
        if V<=0:
            V = math.sqrt(xd_d**2 + yd_d**2) # reset to nominal velocity
        om = (u2*math.cos(th) - u1*math.sin(th))/V

    else:
        # Use pose controller from problem 3
        ctrl = ctrl_pose(x,y,th,x_g,y_g,th_g)
        V = ctrl[0]
        om = ctrl[1]

    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
