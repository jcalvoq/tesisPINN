#LotkaVolterra Equations

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#r = prey , f = predator
t_ini , t_fin = 0,52 #t inicial y t final
f_ini = 20 #condiciones iniciales
#alpha: tasa nacimiento presas(x) ; beta: tasa de interacciones fatales
#gamma: tasa mortalidad depredadores(y) ; delta: tasa de crecimiento por depredacion
alpha , beta , gamma , delta = 0.1 , 0.002 , 0.2 , 0.0025 #parametros del sistema
n0 = 2000
n1 = 2000
n2 = 4500 #numero de puntos
n = n1 + n2
t = torch.linspace(t_ini,t_fin,n,requires_grad = True)[:,None].to(device) #array de n puntos desde t_ini a t_fin
epochs = 40000 #numero de epochs
ls = 120 #numero de nodos
tau = 16
lr = 1e-3

data = ["num puntos=",n,"num epochs=",epochs,"num nodos=",ls,"activacion=","Sigmoid","t_ini y t_fin",t_ini," a ",t_fin,"lr=",lr,"tau=",tau]

vec_loss_R = []
vec_loss_F = []

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
        nn.Sigmoid(),nn.Linear(ls,1))

    #Definir la red
    def forward(self,x):
        output = self.linear_sigmoid_stack(x)
        return output

#Definir una instancia de la clase Network
R = Network()
F = Network()
R = R.to(device)
F = F.to(device)

def IC_R(t):
    return 80

def loss(t_nT_to_0,t_0_to_T,t_T_to_fin , epoch):
    r_ic , r_ltT , r_gtT = R(t_nT_to_0) , R(t_0_to_T) , R(t_T_to_fin)
    f_ltT , f_gtT = F(t_0_to_T) , F(t_T_to_fin)
    
    r_gtT_mT = R(t_T_to_fin - tau)
    
    dr_dt_ltT = torch.autograd.grad(r_ltT.sum() , t_0_to_T , create_graph = True)[0]
    dr_dt_gtT = torch.autograd.grad(r_gtT.sum() , t_T_to_fin , create_graph = True)[0]
    
    df_dt_ltT = torch.autograd.grad(f_ltT.sum() , t_0_to_T , create_graph = True)[0]
    df_dt_gtT = torch.autograd.grad(f_gtT.sum() , t_T_to_fin , create_graph = True)[0]
    
    E_rabbits_IC = (r_ic - IC_R(t_nT_to_0)).pow(2)
    E_rabbits_IC = torch.mean(E_rabbits_IC)
    
    E_foxes_IC = (f_ltT[0] - f_ini).pow(2)
    E_foxes_IC = torch.mean(E_foxes_IC)
    
    E_rabbits_ltT = (dr_dt_ltT - (alpha * IC_R(t_0_to_T - tau)) + (beta * r_ltT * f_ltT)).pow(2)
    E_rabbits_ltT = torch.mean(E_rabbits_ltT)
    
    E_foxes_ltT = (df_dt_ltT + (gamma * f_ltT) - (delta * r_ltT * f_ltT)).pow(2)
    E_foxes_ltT = torch.mean(E_foxes_ltT)
    
    E_rabbits_gtT = (dr_dt_gtT - (alpha * r_gtT_mT) + (beta * r_gtT * f_gtT)).pow(2)
    E_rabbits_gtT = torch.mean(E_rabbits_gtT)
    
    E_foxes_gtT = (df_dt_gtT + (gamma * f_gtT) - (delta * r_gtT * f_gtT)).pow(2)
    E_foxes_gtT = torch.mean(E_foxes_gtT)

    loss_R = (1 / 3) * (E_rabbits_IC + E_rabbits_ltT + E_rabbits_gtT)
    loss_F = (1 / 3) * (E_foxes_IC + E_foxes_ltT + E_foxes_gtT)
    
    if(epoch % 500 == 0):
        vec_loss_R.append(loss_R.detach().cpu().numpy())
        vec_loss_F.append(loss_F.detach().cpu().numpy())

    return loss_R , loss_F

def train():
    optimizer_R = torch.optim.Adam(R.parameters() , lr = lr)
    optimizer_F = torch.optim.Adam(F.parameters() , lr = lr)
    
    nT_to_0 = torch.linspace(-tau,0,n0,requires_grad = True)[:,None].to(device)
    zero_to_T = torch.linspace(0,tau,n1,requires_grad = True)[:,None].to(device)
    T_to_fin = torch.linspace(tau,t_fin,n2,requires_grad = True)[:,None].to(device)
    
    for epoch in range(epochs):
        optimizer_R.zero_grad()
        optimizer_F.zero_grad()
        loss_r , loss_f = loss(nT_to_0 , zero_to_T , T_to_fin , epoch)
        loss_r.backward(retain_graph = True)
        loss_f.backward()
        optimizer_R.step()
        optimizer_F.step()
        if(epoch % 1000 == 0):
            torch.save(R.state_dict() , "/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/solR_w" + str(int(epoch/1000)) + ".pt")
            torch.save(F.state_dict() , "/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/solF_w" + str(int(epoch/1000)) + ".pt")
            
train()

vec_loss_F = np.array(vec_loss_F)
vec_loss_R = np.array(vec_loss_R)

np.savetxt("/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/lossR.txt",vec_loss_R,delimiter=' ')
np.savetxt("/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/lossF.txt",vec_loss_F,delimiter=' ')

with open("/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/datos.txt",'w') as file:
    file.write(str(data))

torch.save(R.state_dict(),"/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/solR_w_final.pt")
torch.save(F.state_dict(),"/LUSTRE/home/jcalvo/Tesis/LVDelay/Weigths/solF_w_final.pt")
