import numpy as np
from utils import wrapToPi
import math

def ctrl_pose(x,y,th,x_g,y_g,th_g):
    #(x,y,th): current state
    #(x_g,y_g,th_g): desired final state

    
    #Code pose controller
    rho = math.sqrt((x-x_g)**2 + (y-y_g)**2);
    alpha = wrapToPi(math.atan2((y-y_g), (x-x_g)) - (th) + math.pi);
    delta = wrapToPi(alpha + (th-th_g));

    #Define control inputs (V,om) - without saturation constraints
    k1 = 0.3
    k2 = 0.3
    k3 = 0.3
    V = k1*rho*math.cos(alpha)
    om = k2*alpha + k1*math.cos(alpha)*np.sinc(alpha/math.pi)*(alpha + k3*delta)

    # Apply saturation limits
    V = np.sign(V)*min(0.5, np.abs(V))
    om = np.sign(om)*min(1, np.abs(om))

    return np.array([V, om])
