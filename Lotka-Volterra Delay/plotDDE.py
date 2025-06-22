import torch
import torch.nn as nn
#from torch.autograd import grad 
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
import time
from LVDelayEuler import Euler
'''
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22
'''
ls = 120 #Numero de nodos en la red

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

nnmodelR = Network()
nnmodelF = Network()

#parameters for plotting in the regular mesh
Nt=500

#domain parameters
t0 = 0.0
tf = 52.0
n = 6500
f_ini = 20.0

tau = 16

dt = (tf - t0)/n

eul_t , eul_x , eul_y = Euler(t0 , tf , f_ini , dt , tau)

# Plotting axis
sst=(Nt,1)
ta=np.zeros(sst)
h=(tf-t0)/float(Nt)
for z in range(Nt):
    ta[z,0]=t0+z*h
t= Variable(torch.from_numpy(ta).float(),requires_grad=True) 

#Here I will load the weights I made in a previous run.
NN = 40
 #manual entry/number of plots


for j in range(NN):
    nombre = str(j)
    nnmodelR.load_state_dict(torch.load("solR_w"+nombre+".pt",weights_only=False) ,strict = False)
    nnmodelF.load_state_dict(torch.load("solF_w"+nombre+".pt",weights_only=False) ,strict = False)
    print("sol_w_"+nombre+".pt")
    nnmodelR.eval()
    nnmodelF.eval()

    rabbits = nnmodelR(t)
    foxes = nnmodelF(t)  
    
    rabbits = rabbits.detach().cpu().numpy()
    foxes = foxes.detach().cpu().numpy()
    
    tt = t
    tp = tt.data.cpu().numpy()
    
    nombre2 = str(j * 1000)
    
    plt.figure(figsize = (10,6))
    plt.plot(tp,rabbits,label = "x(t) (Predicted)")
    plt.plot(eul_t,eul_x,label = "x(t) (Exact)")
    plt.plot(tp,foxes,label = "y(t) (Predicted)") 
    plt.plot(eul_t,eul_y,label = "y(t) (Exact)") 
    plt.xlabel('t')
    plt.legend(loc = 'best' , fontsize = 19)
    plt.title("Epoch " + nombre2)
    plt.grid()
    if (j % 5 == 0):
        plt.savefig("LVDEpoch" + nombre2,bbox_inches = 'tight')
    plt.show()
    
    time.sleep(0.5)    # Pause 5.5 seconds
    
nnmodelR.load_state_dict(torch.load("solR_w_final.pt",weights_only=False),strict = False)
nnmodelR.eval()
nnmodelF.load_state_dict(torch.load("solF_w_final.pt",weights_only=False),strict = False)
nnmodelF.eval()
rabbits = nnmodelR(t)  
foxes = nnmodelF(t)  

rabbits = rabbits.detach().cpu().numpy()
foxes = foxes.detach().cpu().numpy()
    
tt = t
tp = tt.data.cpu().numpy()
           
plt.figure(figsize = (10,6))
plt.plot(tp,rabbits,label = "x(t) (Predicted)")
plt.plot(eul_t,eul_x,label = "x(t) (Exact)")
plt.plot(tp,foxes,label = "y(t) (Predicted)") 
plt.plot(eul_t,eul_y,label = "y(t) (Exact)") 
plt.xlabel('t')
plt.legend(loc = 'best', fontsize = 19)
plt.title("Epoch 40000")
plt.grid()
plt.savefig("LVDFinalSolution",bbox_inches = 'tight')
plt.show()