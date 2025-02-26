import numpy as np
from sys import argv
import matplotlib.pyplot as plt

# Coordinate system
#   .---> x
#   |
# y v



if __name__ == '__main__':
    #seed = int(argv[1])
    index = 1
    
    seeds = [i for i in range(750,1500,5)] #Seeds of the simulations
    t = 500                             #Time steps of the simulations
    tstep = 1
    
    height = 97  # Height of the image in pixels
    width = 97   # Width of the image in pixels
    
    piv_x = 48        # pivot x coord 
    piv_y = 48        # pivot y coord 
    l = 9.8 * 2       # length of the pendulum (equal to g)
    A = 2.0           # decay of the soft circle (logistic function)
    R = 3.0           # Radius of the soft circle
    e = np.exp(-A*R)
    a = 1 + e**2
    b = 2*e
    lbA = l*b*A
    

    x = np.linspace(0,width,width)
    y = np.linspace(0,height,height)
    data = np.empty([len(seeds),int(t/tstep),len(x),len(y)],dtype=np.float32)
    data2 = np.empty([len(seeds),int(t/tstep),len(x),len(y)],dtype=np.float32)
    data3 = np.empty([len(seeds),int(t/tstep),len(x),len(y)],dtype=np.float32)
    label = np.empty([len(seeds),int(t/tstep)],dtype=np.float32)
    
    for idx in range(len(seeds)):
        theta = []
        omega = []
        with open('sims/sim_output_{0:04d}.dat'.format(seeds[idx]), 'r') as ifile:
            for line in ifile:
                v = list(map(float, line.split()))
                theta.append(v[1])
                omega.append(v[2])

        temp = []
        temp_d = []
        temp_dd = []
        lab = []
        for n in range(0,t,tstep):
            lab.append(theta[n])
            
            #print(idx,n)
            mat = np.zeros((height, width))
            mat_d = np.zeros((height, width))
            mat_dd = np.zeros((height, width))
            
            # Generate gaussian around pendulum position
            xx, yy = np.meshgrid(x,y)

            pos_x = piv_x + l*np.sin(theta[n])
            pos_y = piv_y + l*np.cos(theta[n])
            D = np.sqrt((xx-pos_x)**2+(yy-pos_y)**2)
            mat = 1.0 / (a + b*np.cosh(A*D))
            temp.append(mat)

            # Time derivative of the previous function
            c1 = np.cos(theta[n])
            s1 = np.sin(theta[n])
            cs1 = (xx-pos_x)*c1 - (yy-pos_y)*s1
            sh1 = np.sinh(A*D)
            mat_d = (lbA/D)*sh1*cs1*omega[n]*mat**2
            temp_d.append(mat_d)

            # 2nd time derivative
            cs2 = (xx-pos_x)*s1 + (yy-pos_y)*c1
            a1 = mat*l*omega[n]**2/D**2
            a2 = 2*mat_d
            a3 = -mat*np.sin(theta[n]) # dd_theta=-sin(theta) Newton's 2nd law
            a4 = A/np.tanh(A*D)
            a5 = 1.0/cs1
            mat_dd = (lbA/D)*sh1*cs1*mat * (a1+a2+a3-(a4+a5)*cs2*omega[n]**2*mat)
            temp_dd.append(mat_dd)
            
            # plt.imshow(mat,cmap='grey')
            # plt.savefig('images/sample/pos/pos_{0:04d}_{1:05d}.jpg'.format(seeds[idx],n))
            # plt.imshow(mat_d,cmap='grey')
            # plt.savefig('images/sample/vel/vel_{0:04d}_{1:05d}.jpg'.format(seeds[idx],n))
            # plt.imshow(mat_dd,cmap='grey')
            # plt.savefig('images/sample/acel/acel_{0:04d}_{1:05d}.jpg'.format(seeds[idx],n))

        data[idx] = np.array(temp)
        data2[idx] = np.array(temp_d)
        data3[idx] = np.array(temp_dd)
        label[idx] = np.array(lab)

        print(label[idx].shape,data[idx].shape,data2[idx].shape,data3[idx].shape)

    with open(f'X_{index}.npy', 'wb') as f:
        np.save(f,data)
    del data
    with open(f'Xdot_{index}.npy', 'wb') as f:
        np.save(f,data2)
    del data2
    with open(f'Xddot_{index}.npy', 'wb') as f:
        np.save(f,data3)
    del data3
    with open(f'Y_{index}.npy', 'wb') as f:
        np.save(f,label)
    del label
    
