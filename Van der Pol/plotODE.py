import torch
import torch.nn as nn
#from torch.autograd import grad 
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
import time
from RK4 import vanDerPol

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22


ls = 60 #Numero de nodos en la red


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

nnmodel = Network()

#parameters for plotting in the regular mesh
Nt=500

#domain parameters
t0=0.0
tf=20.0
n= 20000

# Plotting axis
sst=(Nt,1)
ta=np.zeros(sst)
h=(tf-t0)/float(Nt)
for z in range(Nt):
    ta[z,0]=t0+z*h
t= Variable(torch.from_numpy(ta).float(),requires_grad=True) 

#Here I will load the weights I made in a previous run.
NN = 50 #manual entry/number of plots

np_t = torch.linspace(t0,tf,n)
rk_t = np_t.detach().numpy()
vec_x,vec_y = vanDerPol(rk_t,1.0,0.0,0.001)


for j in range(NN):
    nombre = str(j)
    nnmodel.load_state_dict(torch.load("sol_w"+nombre+".pt",weights_only=False))
    print("sol_w_"+nombre+".pt")
    nnmodel.eval()

    outpaux = nnmodel(t)  
    outpu1=outpaux[:,0]

    tt=t
    tp=tt.data.cpu().numpy()
    up=outpu1.data.cpu().numpy()
    nombre2 = str(j * 1000)
    plt.figure(figsize = (10,6))
    plt.plot(tp,up,label = "Predicted")
    plt.plot(rk_t,vec_x,label = "Exact")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(loc = 'upper left' , fontsize = 19)
    plt.title("Epoch " + nombre2)
    plt.grid()
    if (j % 10 == 0):
        plt.savefig("VDPEpoch"+nombre+"000",bbox_inches = 'tight')
    plt.show()
    
    time.sleep(0.5)    # Pause 5.5 seconds
    
nnmodel.load_state_dict(torch.load("sol_w_final.pt",weights_only=False))
nnmodel.eval()
outpaux = nnmodel(t)  
outpu1=outpaux[:,0]

tt=t
tp=tt.data.cpu().numpy()
up=outpu1.data.cpu().numpy()
           
plt.figure(figsize = (10,6))
plt.plot(tp,up,label = "Predicted")
plt.plot(rk_t,vec_x,label = "Exact")
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(loc = 'upper left', fontsize = 19)
plt.title("Epoch 50000")
plt.grid()
plt.savefig("VDPEpoch50000",bbox_inches = 'tight')
plt.show()