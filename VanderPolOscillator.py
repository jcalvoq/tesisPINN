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
mu = 0.25
n = 20000 #numero de puntos
epochs = 100000 #numero de epochs
tt = torch.linspace(t_ini,t_fin,n)[:,None] #array de n puntos desde t_ini a t_fin
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
        self.linear_sigmoid_stack = nn.Sequential(nn.Linear(1,ls),nn.Sigmoid(),nn.Linear(ls,ls),nn.Sigmoid(),nn.Linear(ls,ls),nn.Sigmoid(),nn.Linear(ls,ls),nn.Sigmoid(),nn.Linear(ls,1))
        self.linear_tanh_stack = nn.Sequential(nn.Linear(1,ls),nn.Tanh(),nn.Linear(ls,ls),nn.Tanh(),nn.Linear(ls,ls),nn.Tanh(),nn.Linear(ls,ls),nn.Tanh(),nn.Linear(ls,1))
        
    #Definir la red
    def forward(self,x):
        #output = self.linear_sigmoid_stack(x)
        output = self.linear_tanh_stack(x)
        return output

#Crea una instancia de la clase Network        
N = Network()
N = N.to(device)
     
#Definir la funcion de perdida
def loss(x):
    y = N(x)
    dy_dx = torch.autograd.grad(y.sum() , x , create_graph = True)[0]   
    dy_dx2 = torch.autograd.grad(dy_dx.sum() , x , create_graph = True)[0]   
    
    E_ode = (dy_dx2 - mu*(1 - torch.pow(y,2))*dy_dx + y).pow(2)
    E_ode = torch.mean(E_ode)
    
    E_ic = (y[0] - u_ini).pow(2)
    E_ic = torch.mean(E_ic)
    
    loss = (1/(n+1))*(E_ode + E_ic)
    vec_loss.append(loss.detach().numpy())
    
    return loss

def train():
    optimizer = torch.optim.Adam(N.parameters(),lr)
    t0 = torch.tensor([t_ini] , requires_grad = True)
    t0 = torch.unsqueeze(t0,1)
    
    for epoch in range(epochs):
        optimizer.zero_grad() #Regresa todo a 0
        t = torch.tensor(np.random.uniform(t_ini, t_fin, (n, 1)), dtype=torch.float32, requires_grad=True) #Crea un tensor de t_ini a t_fin con n puntos random
        t = torch.cat((t0,t))
        l = loss(t)
        l.backward()
        optimizer.step()
               
        if (epoch % 1000) == 0:    
            torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/WeightsLoss/sol_w" + str(int(epoch/1000)) + ".pt")
    return N

M = train()

with torch.no_grad():
    uu = M.forward(tt)

vec_loss = np.array(vec_loss)
np.savetxt("/LUSTRE/home/jcalvo/Tesis/WeightsLoss/loss.txt",vec_loss,delimiter=' ')

with open("/LUSTRE/home/jcalvo/Tesis/WeightsLoss/datos.txt",'w') as file:
    file.write(str(data))

torch.save(N.state_dict(),"/LUSTRE/home/jcalvo/Tesis/WeightsLoss/sol_w_final.pt")
