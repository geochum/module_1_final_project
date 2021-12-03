import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GRAVITATIONAL_CONSTANT = 6.67408e-11

def explicit_euler(F, Y0, h, n):
    Y = []
    Y.append(Y0)
    for k in range(n - 1):
        Y.append(Y[k] + h * F(Y[k][0],Y[k][1]))
    return Y

def runge_kutta(F, Y0, h, n):
    Y = []
    Y.append(Y0)
    for k in range(n - 1):
        mk = F(*Y[k]) # (Forward) Euler
        nk = F(*(Y[k] + mk * h / 2)) # Midpoint slope
        pk = F(*(Y[k] + nk * h / 2)) # Better midpoint slope
        qk = F(*(Y[k] + pk * h)) # Endpoint slope
        Y.append(Y[k] + (h / 6) * (mk + 2 * nk + 2 * pk + qk))
    return Y

def conservative(F, G, Y0, h, n):
    TOL = 1e-9 # tolerance for fixed point iteration
    Y = []
    Y.append(Y0)
    Y.append(fixed_point(G, Y[0], Y[0] + h * F(Y[0][0],Y[0][1]), h, TOL))
    for k in range(n - 2):
        Y.append(fixed_point(G, Y[k], Y[k+1], h, TOL))
    return Y

def fixed_point(G, Y0, Y1, h, TOL):
    temp = G(Y0[0],Y0[1],Y1[0],Y1[1],h)
    if(np.linalg.norm(temp-Y1) < TOL):
        return temp
    else:
        return fixed_point(G, Y0, temp, h, TOL)
    
def main():
    ## t-interval [t_min,t_max] with step size h
    #t_min, t_max, h = 0, 8.094721, 0.01
    
    #name = "Ovals with flourishes"
    
    ## Initial positions
    #r_1_x, r_1_y, r_1_z = 0.716248295713, 0.384288553041, 0
    #r_2_x, r_2_y, r_2_z = 0.086172594591, 1.342795868577, 0
    #r_3_x, r_3_y, r_3_z = 0.538777980808, 0.481049882656, 0
    
    ## Initial velocities
    #v_1_x, v_1_y, v_1_z = 1.245268230896, 2.444311951777, 0
    #v_2_x, v_2_y, v_2_z = -0.675224323690, -0.962879613630, 0
    #v_3_x, v_3_y, v_3_z = -0.570043907206, -1.481432338147, 0
    
    ########################################
    
    ## t-interval [t_min,t_max] with step size h
    #t_min, t_max, h = 0, 6.324449, 0.01 
    
    #name = "Figure 8"
    #p1, p2 = 0.347111, 0.532728
    
    ## Initial positions
    #r_1_x, r_1_y, r_1_z = -1, 0, 0
    #r_2_x, r_2_y, r_2_z = 1, 0, 0
    #r_3_x, r_3_y, r_3_z = 0, 0, 0
    
    ## Initial velocities
    #v_1_x, v_1_y, v_1_z = p1, p2, 0
    #v_2_x, v_2_y, v_2_z = p1, p2, 0
    #v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0
    
    ########################################
    
    ## t-interval [t_min,t_max] with step size h
    #t_min, t_max, h = 0, 21.270975, 0.0001
    
    #name = "Dragonfly"
    #p1, p2 = 0.080584, 0.588836
    
    ## Initial positions
    #r_1_x, r_1_y, r_1_z = -1, 0, 0
    #r_2_x, r_2_y, r_2_z = 1, 0, 0
    #r_3_x, r_3_y, r_3_z = 0, 0, 0
    
    ## Initial velocities
    #v_1_x, v_1_y, v_1_z = p1, p2, 0
    #v_2_x, v_2_y, v_2_z = p1, p2, 0
    #v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0
    
    ########################################
    
    ## t-interval [t_min,t_max] with step size h
    #t_min, t_max, h = 0, 10.962563, 0.00001   
    
    #name = "Yin-Yang 1b"
    #p1, p2 = 0.282699, 0.327209
    
    ## Initial positions
    #r_1_x, r_1_y, r_1_z = -1, 0, 0
    #r_2_x, r_2_y, r_2_z = 1, 0, 0
    #r_3_x, r_3_y, r_3_z = 0, 0, 0
    
    ## Initial velocities
    #v_1_x, v_1_y, v_1_z = p1, p2, 0
    #v_2_x, v_2_y, v_2_z = p1, p2, 0
    #v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0
    
    ########################################
    
    ## t-interval [t_min,t_max] with step size h
    #t_min, t_max, h = 0, 17.328370, 0.0001  
    
    #name = "Yin-Yang 1a"
    #p1, p2 = 0.513938, 0.304736
    
    ## Initial positions
    #r_1_x, r_1_y, r_1_z = -1, 0, 0
    #r_2_x, r_2_y, r_2_z = 1, 0, 0
    #r_3_x, r_3_y, r_3_z = 0, 0, 0
    
    ## Initial velocities
    #v_1_x, v_1_y, v_1_z = p1, p2, 0
    #v_2_x, v_2_y, v_2_z = p1, p2, 0
    #v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0
    
    ########################################
    
    # t-interval [t_min,t_max] with step size h
    t_min, t_max, h = 0, 55.501762, 0.00001    
    
    name = "Yarn"
    p1, p2 = 0.559064, 0.349192
    
    # Initial positions
    r_1_x, r_1_y, r_1_z = -1, 0, 0
    r_2_x, r_2_y, r_2_z = 1, 0, 0
    r_3_x, r_3_y, r_3_z = 0, 0, 0
    
    # Initial velocities
    v_1_x, v_1_y, v_1_z = p1, p2, 0
    v_2_x, v_2_y, v_2_z = p1, p2, 0
    v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0
    
    ########################################
    
    ## t-interval [t_min,t_max] with step size h
    #t_min, t_max, h = 0, 40, 0.000001    
    
    #name = "BUTTERFLY IV"
    #p1, p2 = 0.350112, 0.079339
    
    ## Initial positions
    #r_1_x, r_1_y, r_1_z = -1, 0, 0
    #r_2_x, r_2_y, r_2_z = 1, 0, 0
    #r_3_x, r_3_y, r_3_z = 0, 0, 0
    
    ## Initial velocities
    #v_1_x, v_1_y, v_1_z = p1, p2, 0
    #v_2_x, v_2_y, v_2_z = p1, p2, 0
    #v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0

    ########################################
    
    # number of iterations
    n = int((t_max - t_min) / h) + 1
    
    # Constants
    G, m_1, m_2, m_3 = 1, 1, 1, 1
    
    # list of t values
    T = np.linspace(t_min, t_max, n) 
    
    # Initial conditions
    U0 = np.array([r_1_x,v_1_x,r_1_y,v_1_y,r_1_z,v_1_z,r_2_x,v_2_x,r_2_y,v_2_y,r_2_z,v_2_z,r_3_x,v_3_x,r_3_y,v_3_y,r_3_z,v_3_z])    
    
    v_1 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_2 * (u_7 - u_1) / ((u_7 - u_1) ** 2 + (u_9 - u_3) ** 2 + (u_11 - u_5) ** 2) ** (3/2) + \
             m_3 * (u_13 - u_1) / ((u_13 - u_1) ** 2 + (u_15 - u_3) ** 2 + (u_17 - u_5) ** 2) ** (3/2))
    v_2 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_2 * (u_9 - u_3) / ((u_7 - u_1) ** 2 + (u_9 - u_3) ** 2 + (u_11 - u_5) ** 2) ** (3/2) + \
             m_3 * (u_15 - u_3) / ((u_13 - u_1) ** 2 + (u_15 - u_3) ** 2 + (u_17 - u_5) ** 2) ** (3/2))
    v_3 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_2 * (u_11 - u_5) / ((u_7 - u_1) ** 2 + (u_9 - u_3) ** 2 + (u_11 - u_5) ** 2) ** (3/2) + \
             m_3 * (u_17 - u_5) / ((u_13 - u_1) ** 2 + (u_15 - u_3) ** 2 + (u_17 - u_5) ** 2) ** (3/2)) 
    v_4 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_1 - u_7) / ((u_1 - u_7) ** 2 + (u_3 - u_9) ** 2 + (u_5 - u_11) ** 2) ** (3/2) + \
             m_3 * (u_13 - u_7) / ((u_13 - u_7) ** 2 + (u_15 - u_9) ** 2 + (u_17 - u_11) ** 2) ** (3/2))
    v_5 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_3 - u_9) / ((u_1 - u_7) ** 2 + (u_3 - u_9) ** 2 + (u_5 - u_11) ** 2) ** (3/2) + \
             m_3 * (u_15 - u_9) / ((u_13 - u_7) ** 2 + (u_15 - u_9) ** 2 + (u_17 - u_11) ** 2) ** (3/2))
    v_6 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_5 - u_11) / ((u_1 - u_7) ** 2 + (u_3 - u_9) ** 2 + (u_5 - u_11) ** 2) ** (3/2) + \
             m_3 * (u_17 - u_11) / ((u_13 - u_7) ** 2 + (u_15 - u_9) ** 2 + (u_17 - u_11) ** 2) ** (3/2))
    v_7 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_1 - u_13) / ((u_1 - u_13) ** 2 + (u_3 - u_15) ** 2 + (u_5 - u_17) ** 2) ** (3/2) + \
             m_2 * (u_7 - u_13) / ((u_7 - u_13) ** 2 + (u_9 - u_15) ** 2 + (u_11 - u_17) ** 2) ** (3/2))
    v_8 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_3 - u_15) / ((u_1 - u_13) ** 2 + (u_3 - u_15) ** 2 + (u_5 - u_17) ** 2) ** (3/2) + \
             m_2 * (u_9 - u_15) / ((u_7 - u_13) ** 2 + (u_9 - u_15) ** 2 + (u_11 - u_17) ** 2) ** (3/2))
    v_9 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_5 - u_17) / ((u_1 - u_13) ** 2 + (u_3 - u_15) ** 2 + (u_5 - u_17) ** 2) ** (3/2) + \
             m_2 * (u_11 - u_17) / ((u_7 - u_13) ** 2 + (u_9 - u_15) ** 2 + (u_11 - u_17) ** 2) ** (3/2))
    
    F = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        np.array([u_2, v_1(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_4, v_2(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_6, v_3(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_8, v_4(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_10, v_5(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_12, v_6(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_14, v_7(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_16, v_8(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_18, v_9(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18)])
    
    U = runge_kutta(F, U0, h, n)
    
    R1 = []
    R2 = []
    R3 = []
    for u in U:
        R1.append([u[index] for index in [0, 2, 4]])
        R2.append([u[index] for index in [6, 8, 10]])
        R3.append([u[index] for index in [12, 14, 16]])
    
    # Runge-Kutta Plot
    
    X1 = [p[0] for p in R1]
    Y1 = [p[1] for p in R1]
    Z1 = [p[2] for p in R1]
    
    X2 = [p[0] for p in R2]
    Y2 = [p[1] for p in R2]
    Z2 = [p[2] for p in R2]
    
    X3 = [p[0] for p in R3]
    Y3 = [p[1] for p in R3]
    Z3 = [p[2] for p in R3]    
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection ='3d')
    ax.plot3D(X1, Y1, Z1, 'blue', label = 'Mass 1')
    ax.plot3D(X2, Y2, Z2, 'red', label = 'Mass 2')
    ax.plot3D(X3, Y3, Z3, 'green', label = 'Mass 3')
    ax.legend()
    ax.set_title(name)
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20) 
    plt.show()

if __name__ == '__main__':
    main()