#Importar las librerias necesarias
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(7)

#Usar GPU si esta disponible, en otro caso usar CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Definicion de constantes globales
t_ini = 0.0 #t inicial
t_fin = 12.0 #t final
y_ini = 0.01 #u inicial
n = 1000 #numero de puntos
epochs = 200000 #numero de epochs
tt = torch.linspace(t_ini,t_fin,n)[:,None]
tt = tt.to(device) #array de n puntos desde t_ini a t_fin
ls = 40 #Numero de nodos en la red
lr = 1e-2
data = ["num puntos=",n,"num epochs=",epochs,"num nodos=",ls,"activacion=","sigmoid","t_ini y t_fin",t_ini," a ",t_fin,"lr=",lr]

#Definicion de variables globales 
vec_loss = [] #array para guardar los valores de la loss function
vec_error = []

np_t = tt.detach().cpu().numpy()
exact = (2 * torch.exp(tt)) / (torch.exp(tt) + 199)

class Network(nn.Module):
    #Definir el constructor de la clase
    def __init__(self):
        super().__init__()
        #Definir la red
        self.linear_sigmoid_stack = nn.Sequential(nn.Linear(1,ls),
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
     
#Definir la funcion de perdida
def loss(x,epoch):
    y = N(x)
    dy_dx = torch.autograd.grad(y.sum() , x , create_graph = True)[0]   
    
    E_ode = (dy_dx - y * (1 - (y / 2))).pow(2)
    E_ode = torch.mean(E_ode)
    
    E_ic = (y[0] - y_ini).pow(2)
    E_ic = torch.mean(E_ic)

    if(epoch <= 50000):
      E_add = (y[n - 1] - 1.0).pow(2)
      E_add = torch.mean(E_add)
    else:
      E_add = 0
    
    loss = (1/(n+1))*(E_ode + E_ic + E_add)
    
    if(epoch % 500) == 0:          
        error = (y - exact).pow(2)
        error = torch.mean(error)
        
        vec_loss.append(loss.detach().cpu().numpy())
        vec_error.append(error.detach().cpu().numpy())
    
    return loss

def train():
    optimizer = torch.optim.Adam(N.parameters(),lr = lr)
    t = torch.linspace(t_ini , t_fin , n , requires_grad = True)[:,None].to(device)
    for epoch in range(epochs):
        optimizer.zero_grad() #Regresa todo a 0
        l = loss(t,epoch)
        l.backward()
        optimizer.step()
               
        if (epoch % 1000) == 0:    
            torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/Logstic/Weights/sol_w" + str(int(epoch/1000)) + ".pt")
            
train()

vec_loss = np.array(vec_loss)
np.savetxt("/LUSTRE/home/jcalvo/Tesis/Logstic/Weights/loss.txt",vec_loss,delimiter=' ')
vec_error = np.array(vec_error)
np.savetxt("/LUSTRE/home/jcalvo/Tesis/Logstic/Weights/error.txt",vec_error,delimiter=' ')

with open("/LUSTRE/home/jcalvo/Tesis/Logstic/Weights/datos.txt",'w') as file:
    file.write(str(data))

torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/Logstic/Weights/sol_w_final.pt")

#exact = (2 * np.exp(tt)) / (np.exp(tt) + 199)
