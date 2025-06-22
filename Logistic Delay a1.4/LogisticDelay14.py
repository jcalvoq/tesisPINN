#Importar las librerias necesarias
import torch
import torch.nn as nn
import numpy as np
from EulerDelay import Euler

torch.manual_seed(11)

#Usar GPU si esta disponible, en otro caso usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Definicion de constantes globales
t_ini = 0.0 #t inicial
t_fin = 20.0 #t final
tau = 1.0 #valor de tau (retardo)
n0 = 250
n1 = 250
n2 = 5000 #numero de puntos
n = n1 + n2
epochs = 140001 #numero de epochs
a = 1.4
ls = 25 #Numero de nodos en la red
lr = 1e-2
tt = torch.linspace(t_ini,t_fin,n)[:,None].to(device)

dt = (t_fin - t_ini)/n
vec_t , vec_y = Euler(t_ini , t_fin , dt , tau , a)
exact = torch.from_numpy(vec_y).to(device)

data = ["a=",a,"num puntos=",n,"num epochs=",epochs,"num nodos=",ls,"activacion=","sigmoid","t_ini y t_fin",t_ini," a ",t_fin,"lr=",lr,"tau=",tau]

#Definicion de variables globales 
vec_loss = [] #array para guardar los valores de la loss function
vec_error = []

class Network(nn.Module):
    #Definir el constructor de la clase
    def __init__(self):
        super().__init__()
        #Definir la red
        self.linear_sigmoid_stack = nn.Sequential(nn.Linear(1,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,1))        
    #Definir la red
    def forward(self,x):
        output = self.linear_sigmoid_stack(x)
        return output

#Crea una instancia de la clase Network        
N = Network()
N = N.to(device)

def IC(t):
    return 0.1

#Definir la funcion de perdida
def loss(t_nT_to_0,t_0_to_T,t_T_to_fin,epoch):
    #Obtener el output de la red en los intervalos (-tau,0),(0,tau),(tau,t_fin), respectivamente
    y_ic , y_ltT , y_gtT = N(t_nT_to_0) , N(t_0_to_T) , N(t_T_to_fin)
    #Obtener el output de la red en el intervalo (tau,t_fin) pero restando tau
    y_gtT_mT = N(t_T_to_fin - tau)
    #Calcular el autograd para el intervalo (0,tau)
    dy_dt_ltT = torch.autograd.grad(y_ltT.sum() , t_0_to_T , create_graph = True)[0]
    #Calcular el autograd para el intervalo (tau,t_fin)      
    dy_dt_gtT = torch.autograd.grad(y_gtT.sum() , t_T_to_fin , create_graph = True)[0]      

    E_IC = (y_ic - IC(t_nT_to_0)).pow(2) #condicion inicial
    E_IC = torch.mean(E_IC)

    E_ltT = (dy_dt_ltT - a*y_ltT*(1 - IC(t_0_to_T - tau))).pow(2) #error modificando en intervalo 0 a tau
    E_ltT = torch.mean(E_ltT)
    
    E_gtT = (dy_dt_gtT - a*y_gtT*(1 - y_gtT_mT)).pow(2) #error para el intervalo tau a t_fin
    E_gtT = torch.mean(E_gtT)
    
    if(epoch < 40000):
        E_add = (y_gtT[int(n2 - 1)] - 1.0).pow(2)
        E_add = torch.mean(E_add)        
    else: E_add = 0
    
    loss = (1/3)*(E_IC + E_ltT + E_gtT + E_add)
    
    if(epoch % 500 == 0):
        vec_loss.append(loss.detach().cpu().numpy())
    
    return loss

def train():
    optimizer = torch.optim.Adam(N.parameters(),lr)
    
    nT_to_0 = torch.linspace(-tau,0,n0,requires_grad = True)[:,None].to(device)
    zero_to_T = torch.linspace(0,tau,n1,requires_grad = True)[:,None].to(device)
    T_to_fin = torch.linspace(tau,t_fin,n2,requires_grad = True)[:,None].to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad() #Regresa todo a 0      
        l = loss(nT_to_0,zero_to_T,T_to_fin,epoch)
        l.backward()
        optimizer.step()
               
        if (epoch % 1000) == 0:    
            torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/Delay/Weights/sol_w" + str(int(epoch/1000)) + ".pt")

train()
    
vec_loss = np.array(vec_loss)
np.savetxt("/LUSTRE/home/jcalvo/Tesis/Delay/Weights/loss.txt",vec_loss,delimiter=' ')

with open("/LUSTRE/home/jcalvo/Tesis/Delay/Weights/datos.txt",'w') as file:
    file.write(str(data))

torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/Delay/Weights/sol_w_final.pt")
