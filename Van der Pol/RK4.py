import numpy as np

mu = 0.25

#dx/dt = y
def f(x,y,t):
    return y

#dy/dt = mu(1-x^2)y-x
def g(x,y,t):
    return mu*(1 - x**2.0)*y - x

def vanDerPol(val_t,x_ini,y_ini,dt):
    k = dt/2
    
    valores_x = []
    valores_y = []
    
    valores_x.append(x_ini)
    valores_y.append(y_ini)
    
    for i in range(len(val_t) - 1):
        x_1 = valores_x[i]
        y_1 = valores_y[i]

        x_2 = valores_x[i] + k * f(x_1,y_1,val_t[i])
        y_2 = valores_y[i] + k * g(x_1,y_1,val_t[i])

        x_3 = valores_x[i] + k * f(x_2,y_2,val_t[i] + k)
        y_3 = valores_y[i] + k * g(x_2,y_2,val_t[i] + k)

        x_4 = valores_x[i] + dt * f(x_3,y_3,val_t[i] + k)
        y_4 = valores_y[i] + dt * g(x_3,y_3,val_t[i] + k)

        x = valores_x[i] + (dt / 6) * (f(x_1,y_1,val_t[i]) + 2 * f(x_2,y_2,val_t[i] + k) + 2 * f(x_3,y_3,val_t[i] + k) + f(x_4,y_4,val_t[i] + dt))
        y = valores_y[i] + (dt / 6) * (g(x_1,y_1,val_t[i]) + 2 * g(x_2,y_2,val_t[i] + k) + 2 * g(x_3,y_3,val_t[i] + k) + g(x_4,y_4,val_t[i] + dt))

        valores_x.append(x)
        valores_y.append(y)

    
    valores_y = np.array(valores_y)
    valores_x = np.array(valores_x)
    
    return valores_x,valores_y
