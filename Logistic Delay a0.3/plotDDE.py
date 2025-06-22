import torch
import torch.nn as nn
#from torch.autograd import grad 
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
import time
from EulerDelay import Euler

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22


ls = 25 #Numero de nodos en la red

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

nnmodel = Network()

#parameters for plotting in the regular mesh
Nt=500

#domain parameters
t0=0.0
tf=20.0

# Plotting axis
sst=(Nt,1)
ta=np.zeros(sst)
h=(tf-t0)/float(Nt)
for z in range(Nt):
    ta[z,0]=t0+z*h
t= Variable(torch.from_numpy(ta).float(),requires_grad=True) 

#Here I will load the weights I made in a previous run.
NN = 20 #manual entry/number of plots

#constantes euler
tau = 1.0
n = 5250
dt = (tf - t0)/5250
a = 0.3

vec_t , vec_y = Euler(t0,tf,dt,tau,a)


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
               
    nombre2 = str(j*1000)
    plt.figure(figsize = (10,6))
    plt.plot(tp,up,label = "Predicted")
    plt.plot(vec_t,vec_y,label = "Exact")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend(loc = 'best')
    plt.title("Epoch " + nombre2)
    plt.grid()
    if (j % 5 == 0):
        plt.savefig("LDa03Solution"+nombre+"000",bbox_inches = 'tight')
    plt.show()
    
    time.sleep(0.5)    # Pause 5.5 seconds
    
nnmodel.load_state_dict(torch.load("sol_w_final.pt",weights_only=False))
nnmodel.eval()
outpaux = nnmodel(t)  
outpu1=outpaux[:,0]

tt=t
tp=tt.data.cpu().numpy()
up=outpu1.data.cpu().numpy()

NNN = NN*1000           
plt.figure(figsize = (10,6))
plt.plot(tp,up,label = "Predicted")
plt.plot(vec_t,vec_y,label = "Exact")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend(loc = 'best')
plt.title("Epoch " + str(NNN))
plt.grid()
plt.savefig("LDa03FinalSolutiona03",bbox_inches = 'tight')
plt.show()