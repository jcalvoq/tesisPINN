#LotkaVolterra Equations

import torch 
import torch.nn as nn
import numpy as np

torch.manual_seed(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#x = prey , y = predator
t_ini , t_fin = 0,50 #t inicial y t final
x_ini , y_ini = 80,20 #condiciones iniciales
#alpha: tasa nacimiento presas(x) ; beta: tasa de interacciones fatales
#gamma: tasa mortalidad depredadores(y) ; delta: tasa de crecimiento por depredacion
#alpha , beta , gamma , delta = 1.0,0.1,1.0,0.05 #parametros del sistema
alpha , beta , gamma , delta = 0.1 , 0.002 , 0.2 , 0.0025 #parametros del sistema
n = 2000 #numero de puntos
epochs = 100000 #numero de epochs
ls = 120 #numero de nodos
lr = 1e-3

data = ["num puntos=",n,"num epochs=",epochs,"num nodos=",ls,"activacion=","sigmoid","t_ini y t_fin",t_ini," a ",t_fin,"lr=",lr]
vec_loss = []

class Network(nn.Module):
    #Definir el constructor de la clase
    def __init__(self):
        super().__init__()
        #Definir la red
        self.linear_sigmoid_stack = nn.Sequential(nn.Linear(1,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,ls),
        nn.Sigmoid(),nn.Linear(ls,2))
        
    #Definir la red
    def forward(self,x):
        output = self.linear_sigmoid_stack(x)
        return output

#Definir una instancia de la clase Network
N = Network()
N = N.to(device)

def loss(t,epoch):
    pred = N(t)
    x = pred[: , 0:1]
    y = pred[: , 1:2]
    
    dx_dt = torch.autograd.grad(x.sum() , t , create_graph = True)[0]
    dy_dt = torch.autograd.grad(y.sum() , t , create_graph = True)[0]
    
    error1 = (dx_dt - (alpha * x) + (beta * x * y)).pow(2)
    error1 = torch.mean(error1)
    
    error2 = (dy_dt + (gamma * y) - (delta * x * y)).pow(2)
    error2 = torch.mean(error2)
    
    eIC1 = (x[0] - x_ini).pow(2)
    eIC1 = torch.mean(eIC1)
    
    eIC2 = (y[0] - y_ini).pow(2)
    eIC2 = torch.mean(eIC2)
    
    loss = (1/4)*(error1 + error2 + eIC1 + eIC2)    
    
    if(epoch % 500 == 0):
        vec_loss.append(loss.detach().cpu().numpy())
    
    return loss

def train():
    optimizer = torch.optim.Adam(N.parameters(),lr = lr)
    t = torch.linspace(t_ini,t_fin,n,requires_grad = True)[:,None].to(device) #array de n puntos desde t_ini a t_fin
    for epoch in range(epochs):
        optimizer.zero_grad()
        l = loss(t , epoch)
        l.backward()
        optimizer.step()
        if(epoch % 1000 == 0): 
            torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/LV/WeigthsLoss/sol_w" + str(int(epoch/1000)) + ".pt")

train()

vec_loss = np.array(vec_loss)
np.savetxt("/LUSTRE/home/jcalvo/Tesis/LV/WeigthsLoss/loss.txt",vec_loss,delimiter=' ')

with open("/LUSTRE/home/jcalvo/Tesis/LV/WeigthsLoss/datos.txt",'w') as file:
    file.write(str(data))

torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/LV/WeigthsLoss/sol_w_final.pt")
