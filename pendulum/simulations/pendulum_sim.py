# Simulation of a simple pendulum using Runge Kutta 4 degree
# as integrator.
# omega'=-(g/l)*sin(theta)
# theta'=omega

import numpy as np
from sys import argv

def accel(theta):
    return -np.sin(theta)

# w_n+1 = w_n + (1/6)(k1w+2k2w+2k3w+k4w)
# k1w = acc(theta_n)*h
# k2w = acc(theta_n+k1t/2)*h
# k3w = acc(theta_n+k2t/2)*h
# k4w = acc(theta_n+k3t)*h
# t_n+1 = t_n + (1/6)(k1t+2k2t+2k3t+k4t)
# k1t = omega_n*h
# k2t = (omega_n+k1w/2)*h
# k3t = (omega_n+k2w/2)*h
# k4t = (omega_n+k3w)*h
def next_state(theta, omega, h):
    k1t = omega*h
    k1w = accel(theta)*h
    k2t = (omega+k1w/2)*h
    k2w = accel(theta+k1t/2)*h
    k3t = (omega+k2w/2)*h
    k3w = accel(theta+k2t/2)*h
    k4t = (omega+k3w)*h
    k4w = accel(theta+k3t)*h
    next_theta = theta + (k1t+2*k2t+2*k3t+k4t)/6
    next_omega = omega + (k1w+2*k2w+2*k3w+k4w)/6
    return next_theta, next_omega

if __name__ == '__main__':
    seed = int(argv[1])
    np.random.seed(seed)

    # sqrt(g/l) = 1 s^-1
    h = 0.01 # s
    
    N_iter = 500

    theta_0 = np.random.uniform(-np.pi/2,np.pi/2)
    omega_0 = np.random.uniform(-2.1,2.1)
    while (omega_0**2/2-np.cos(theta_0)) >= 1: #Pendulum should not loop
        theta_0 = np.random.uniform(-np.pi/2,np.pi/2)
        omega_0 = np.random.uniform(-2.1,2.1)
    
    theta_list = [theta_0]
    omega_list = [omega_0]
    time = [0]
    
    theta = theta_0
    omega = omega_0
    for i in range(N_iter):
        theta, omega = next_state(theta, omega, h)
            
        theta_list.append(theta)
        omega_list.append(omega)
        time.append((i+1)*h)

    with open("sims/sim_output_{0:04d}.dat".format(seed), "w") as ofile:
        for i in range(len(time)): 
            ofile.write('{} {} {}\n'.format(time[i],theta_list[i],omega_list[i]))
