import numpy as np

alpha , beta , gamma , delta = 0.1 , 0.002 , 0.2 , 0.0025

#dx/dt = alpha*x - beta*x*y prey
def f(x,y,t):
    return alpha*x - beta*x*y

#dy/dt = -gamma*y + delta*x*y predator
def g(x,y,t):
    return -gamma*y + delta*x*y

def euler(dt , t_ini , t_fin ,x_ini, y_ini):
    num_iter = int((t_fin - t_ini) / dt)
    t = t_ini
    
    valores_t = []
    valores_x = []
    valores_y = []

    valores_t.append(t_ini)
    valores_x.append(x_ini)
    valores_y.append(y_ini)
    
    for i in range(num_iter):
        valores_x.append(valores_x[i] + dt * f(valores_x[i],valores_y[i],t))
        valores_y.append(valores_y[i] + dt * g(valores_x[i],valores_y[i],t))
        t += dt
        valores_t.append(t)
        
    valores_t = np.array(valores_t)
    valores_x = np.array(valores_x)
    valores_y = np.array(valores_y)
    

    return valores_t , valores_x , valores_y