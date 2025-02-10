import torch
import torch.nn as nn
#from torch.autograd import grad 
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
import time
from roboRK4 import vanDerPol

ls = 60 #Numero de nodos en la red

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
NN = 100 #manual entry/number of plots

vec_x,vec_y,vec_t = vanDerPol(0.0,20.0,1.0,0.0,0.001)


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
               
    plt.figure(figsize = (10,6))
    plt.plot(tp,up,label = "Predicted")
    plt.plot(vec_t,vec_x,label = "Exact",color="red")
    plt.xlabel('t')
    plt.ylabel('u')
    plt.legend(loc = 'best')
    plt.title("Epoch " + nombre + "000")
    plt.grid()
    if (j % 10 == 0):
        plt.savefig("VDPGraphEpoch"+nombre+"000")
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
plt.plot(vec_t,vec_x,label = "Exact",color="red")
plt.xlabel('t')
plt.ylabel('u')
plt.legend(loc = 'best')
plt.title("Epoch 100000")
plt.grid()
plt.savefig("VDPGraphEpoch100000")
plt.show()