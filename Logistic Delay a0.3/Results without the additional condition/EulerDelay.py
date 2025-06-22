import numpy as np 

#definicion de constantes
def IC(t):
    return 0.1

def f(t,y,y_delay,a):
    return a*y*(1 - y_delay)

def aprox_y(x1,x2,y1,y2,tmT):
    frac = (y2 - y1)/(x2 - x1)
    return y1 + frac*(tmT - x1)

def Euler(t_ini , t_fin , dt , tau,a):
    vec_t = np.arange(t_ini , t_fin + dt , dt)
    vec_y = [IC(t_ini)]
    
    for i in range(1,len(vec_t)):
        t_menos_Tau = vec_t[i] - tau    
           
        if t_menos_Tau <= 0: 
            y_delay = IC(t_menos_Tau)
        else:    
            delay_index = int(np.floor((vec_t[i - 1] - tau) / dt)) 
            y_delay = aprox_y(vec_t[delay_index] , vec_t[delay_index + 1] , vec_y[delay_index] , vec_y[delay_index + 1] , t_menos_Tau) 

        val_act = vec_y[i-1] + dt * f(vec_t[i-1],vec_y[i-1],y_delay,a)
        vec_y.append(val_act)
    
    return vec_t , vec_y