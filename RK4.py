import numpy as np
import matplotlib.pyplot as plt

mu = 0.25
alpha,beta,gamma,delta = 1,0.1,1,0.05

#dx/dt = alpha*x - beta*x*y prey
#dx/dt = y
def f(x,y,t):
    #return alpha * x - beta * x * y
    return y

#dy/dt = -gamma*y + delta*x*y predator 
#dy/dt = mu(1-x^2)y-x
def g(x,y,t):
    #return -gamma * y + delta * x * y
    return mu*(1 - x**2.0)*y - x

def vanDerPol(t_ini,t_fin,x_ini,y_ini,dt):
    num_iter = int((t_fin - t_ini) / dt)
    k = dt/2
    t = t_ini
    
    valores_x = []
    valores_y = []
    valores_t = []
    
    valores_x.append(x_ini)
    valores_y.append(y_ini)
    valores_t.append(t_ini)
    
    for i in range(num_iter):
        x_1 = valores_x[i]
        y_1 = valores_y[i]

        x_2 = valores_x[i] + k * f(x_1,y_1,valores_t[i])
        y_2 = valores_y[i] + k * g(x_1,y_1,valores_t[i])

        x_3 = valores_x[i] + k * f(x_2,y_2,valores_t[i] + k)
        y_3 = valores_y[i] + k * g(x_2,y_2,valores_t[i] + k)

        x_4 = valores_x[i] + dt * f(x_3,y_3,valores_t[i] + k)
        y_4 = valores_y[i] + dt * g(x_3,y_3,valores_t[i] + k)

        x = valores_x[i] + (dt / 6) * (f(x_1,y_1,valores_t[i]) + 2 * f(x_2,y_2,valores_t[i] + k) + 2 * f(x_3,y_3,valores_t[i] + k) + f(x_4,y_4,valores_t[i] + dt))
        y = valores_y[i] + (dt / 6) * (g(x_1,y_1,valores_t[i]) + 2 * g(x_2,y_2,valores_t[i] + k) + 2 * g(x_3,y_3,valores_t[i] + k) + g(x_4,y_4,valores_t[i] + dt))

        valores_x.append(x)
        valores_y.append(y)

        t += dt
        valores_t.append(t)
    
    valores_t = np.array(valores_t)
    valores_y = np.array(valores_y)
    valores_x = np.array(valores_x)
    
    return valores_x,valores_y,valores_t

dt = 0.001
t_ini = 0.0
t_fin = 15.0
x_ini = 1.0
y_ini = 0.0 

vec_x , vec_y , vec_t = vanDerPol(t_ini,t_fin,x_ini,y_ini,dt)

plt.plot(vec_t , vec_x)
#plt.plot(vec_t , vec_y,label = "Predator")
plt.grid()
plt.show()