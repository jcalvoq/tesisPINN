import numpy as np 

#definicion de constantes
alpha , beta , gamma , delta = 0.1 , 0.002 , 0.2 , 0.0025

def IC(t):
    return 80

#dx/dt = alpha*x - beta*x*y prey
def R(t,f,r,r_delay):
    return alpha*r_delay - beta*r*f

#dy/dt = -gamma*y + delta*x*y predator
def F(t,f,r):
    return -gamma*f + delta*r*f

def aprox_R(x1,x2,y1,y2,tmT):
    frac = (y2 - y1)/(x2 - x1)
    return y1 + frac*(tmT - x1)

def Euler(t_ini , t_fin ,F_ini, dt , tau):
    vec_t = np.arange(t_ini , t_fin + dt , dt)
    vec_R = [IC(t_ini)]
    vec_F = [F_ini]
    
    for i in range(1,len(vec_t)):
        t_menos_Tau = vec_t[i] - tau    
           
        if t_menos_Tau <= 0: 
            R_delay = IC(t_menos_Tau)
        else:    
            delay_index = int(np.floor((vec_t[i - 1] - tau) / dt)) 
            R_delay = aprox_R(vec_t[delay_index] , vec_t[delay_index + 1] , vec_R[delay_index] , vec_R[delay_index + 1] , t_menos_Tau) 

        vec_R.append(vec_R[i-1] + dt * R(vec_t[i-1] , vec_F[i-1] , vec_R[i-1] , R_delay))
        vec_F.append(vec_F[i-1] + dt * F(vec_t[i-1] , vec_F[i-1] , vec_R[i-1]))
    
    return vec_t , vec_R , vec_F