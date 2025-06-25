#Importar las librerias necesarias
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(884)

#Usar GPU si esta disponible, en otro caso usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Definicion de constantes globales
t_ini = 0.0 #t inicial
t_fin = 20.0 #t final
u_ini = 1.0 #u inicial
y_ini = 0.0
mu = 0.25
n = 20000 #numero de puntos
epochs = 50001 #numero de epochs
tt = torch.linspace(t_ini,t_fin,n)[:,None].to(device) #array de n puntos desde t_ini a t_fin
e = np.linspace(0,epochs,epochs) #array de epochs puntos desde 0 a epochs
ls = 60 #Numero de nodos en la red
lr = 1e-3
data = ["mu=",mu,"num puntos=",n,"num epochs=",epochs,"num nodos=",ls,"activacion=","tanh","t_ini y t_fin",t_ini," a ",t_fin,"lr=",lr]

#Definicion de variables globales 
vec_loss = [] #array para guardar los valores de la loss function

class Network(nn.Module):
    #Definir el constructor de la clase
    def __init__(self):
        super().__init__()
        #Definir la red
        self.linear_tanh_stack = nn.Sequential(nn.Linear(1,ls),
        nn.Tanh(),nn.Linear(ls,ls),
        nn.Tanh(),nn.Linear(ls,ls),
        nn.Tanh(),nn.Linear(ls,ls),
        nn.Tanh(),nn.Linear(ls,1))
        
    #Definir la red
    def forward(self,x):
        output = self.linear_tanh_stack(x)
        return output

#Crea una instancia de la clase Network        
N = Network()
N = N.to(device)
     
#Definir la funcion de perdida
def loss(x,epoch):
    y = N(x)
    dy_dx = torch.autograd.grad(y.sum() , x , create_graph = True)[0]   
    dy_dx2 = torch.autograd.grad(dy_dx.sum() , x , create_graph = True)[0]   
    
    E_ode = (dy_dx2 - mu*(1 - torch.pow(y,2))*dy_dx + y).pow(2)
    E_ode = torch.mean(E_ode)
    
    E_ic = (y[0] - u_ini).pow(2)
    E_ic = torch.mean(E_ic)
    
    E_2 = (dy_dx[0] - y_ini).pow(2)
    E_2 = torch.mean(E_2)
    
    loss = (1/3)*(E_ode + E_ic + E_2)
    
    if (epoch % 500) == 0:
        vec_loss.append(loss.detach().cpu().numpy())
    
    return loss

def train():
    optimizer = torch.optim.Adam(N.parameters(),lr)
    
    t = torch.linspace(t_ini,t_fin,n,requires_grad = True)[:,None].to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad() #Regresa todo a 0
        l = loss(t,epoch)      
        l.backward()
        optimizer.step()
               
        if (epoch % 1000) == 0:    
            torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/WeightsLoss/sol_w" + str(int(epoch/1000)) + ".pt")

train()

vec_loss = np.array(vec_loss)
np.savetxt("/LUSTRE/home/jcalvo/Tesis/WeightsLoss/loss.txt",vec_loss,delimiter=' ')

with open("/LUSTRE/home/jcalvo/Tesis/WeightsLoss/datos.txt",'w') as file:
    file.write(str(data))

torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/WeightsLoss/sol_w_final.pt")
