#Importar las librerias necesarias y un archivo externo que se encarga de graficar
import torch
import torch.nn as nn
import numpy as np
from aux_functions import plotSolution,plotError,plotFinalSolution,plotFinalError,exactSolution

#Fijamos una semilla
torch.manual_seed(46)

#Usar GPU si esta disponible, en otro caso usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Definicion de constantes globales
t_ini = 0.0 #t inicial
t_fin = 10.0 #t final
u_ini = 0.0 #u inicial, la condicion inicial
n = 5000 #Numero de puntos en el intervalo t_ini a t_fin
epochs = 10000 #Numero de epochs
tt = torch.linspace(t_ini,t_fin,n)[:,None].to(device) #Array de n puntos desde t_ini a t_fin
ls = 40 #Numero de nodos en la red
lr = 1e-3 #Learning rate 
optimizer_choice = "Adam"

#Definicion de variables globales 
vec_loss = [] #Array para guardar los valores de la loss function
vec_error = [] #Array para guardar los valores del error exacto

#Obtenemos la solucion exacta de la ecuacion diferencial
np_t = tt.detach().cpu().numpy()
exact = exactSolution(np_t)
exact = torch.from_numpy(exact).to(device)

#Definimos la clase red 
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
def loss(t):
    y = N(t)
    dy_dx = torch.autograd.grad(y.sum() , t , create_graph = True)[0]   
    
    E_ode = (dy_dx + 1/5*y - torch.exp(-t/5)*torch.cos(t)).pow(2)
    E_ode = torch.mean(E_ode)
    
    E_ic = (y[0] - u_ini).pow(2)
    E_ic = torch.mean(E_ic)
    
    loss = (1/(n+1))*(E_ode + E_ic)
        
    error = (y - exact).pow(2)
    error = torch.mean(error)
    vec_loss.append(loss.detach().cpu().numpy())
    vec_error.append(error.detach().cpu().numpy())

    return loss

def train():
    optimizer = torch.optim.Adam(N.parameters(),lr = lr)
    
    t0 = torch.tensor([t_ini] , requires_grad = True, device = device)
    t0 = torch.unsqueeze(t0,1)
    
    t = torch.linspace(t_ini , t_fin , n - 1 ,requires_grad = True, device = device)[:,None]
    t = torch.cat((t0,t))
    
    for epoch in range(epochs):
        optimizer.zero_grad() #Regresa todo a 0
        l = loss(t)
        l.backward()
        optimizer.step()
               
        if (epoch % 1000) == 0:    
            print("Epoch:" , epoch)
            plotSolution(epoch,N,tt,optimizer_choice)
            plotError(len(vec_error),vec_loss,vec_error,optimizer_choice)
            torch.save(N.state_dict(),"D:/Tesis/" + optimizer_choice + "/sol_w_" + str(int(epoch/1000)) + ".pt")

train()

with torch.no_grad():
    uu = N.forward(tt)

xx = uu.detach().cpu().numpy()
tt = tt.detach().cpu().numpy()

#Graficar
plotFinalSolution(tt,xx,optimizer_choice)
plotFinalError(len(vec_loss),vec_loss,vec_error,optimizer_choice)
torch.save(N.state_dict(),"D:/Tesis/" + optimizer_choice +"/sol_w_final.pt")