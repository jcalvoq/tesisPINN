import torch
import torch.nn as nn
#from torch.autograd import grad 
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
import time
from LVEuler import euler

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22


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
        nn.Sigmoid(),nn.Linear(ls,2))
    #Definir la red
    def forward(self,x):
        output = self.linear_sigmoid_stack(x)
        return output

nnmodel = Network()

#parameters for plotting in the regular mesh
Nt=500

#domain parameters
t0=0.0
tf=50.0
n= 1000
x_ini = 80
y_ini = 20

dt = (tf - t0)/n

eul_t , eul_x , eul_y = euler(dt , t0 , tf , x_ini , y_ini)

# Plotting axis
sst=(Nt,1)
ta=np.zeros(sst)
h=(tf-t0)/float(Nt)
for z in range(Nt):
    ta[z,0]=t0+z*h
t= Variable(torch.from_numpy(ta).float(),requires_grad=True) 

#Here I will load the weights I made in a previous run.
NN = 100
 #manual entry/number of plots


for j in range(NN):
    nombre = str(j)
    nnmodel.load_state_dict(torch.load("sol_w"+nombre+".pt",weights_only=False) ,strict = False)
    print("sol_w_"+nombre+".pt")
    nnmodel.eval()

    outpaux = nnmodel(t)  
    
    xx = outpaux[: , 0:1]
    yy = outpaux[: , 1:2]
    
    xx = xx.detach().cpu().numpy()
    yy = yy.detach().cpu().numpy()
    
    tt = t
    tp = tt.data.cpu().numpy()
    
    nombre2 = str(j * 1000)
    
    plt.figure(figsize = (10,6))
    plt.plot(tp,xx,label = "x(t) (Predicted)")
    plt.plot(eul_t,eul_x,label = "x(t) (Exact)")
    plt.plot(tp,yy,label = "y(t) (Predicted)") 
    plt.plot(eul_t,eul_y,label = "y(t) (Exact)") 
    plt.xlabel('t')
    plt.legend(loc = 'best' , fontsize = 19)
    plt.title("Epoch " + nombre2)
    plt.grid()
    if (j % 10 == 0):
        plt.savefig("LVND"+nombre2,bbox_inches = 'tight')
    plt.show()
    
    time.sleep(0.5)    # Pause 5.5 seconds
    
nnmodel.load_state_dict(torch.load("sol_w_final.pt",weights_only=False),strict = False)
nnmodel.eval()
outpaux = nnmodel(t)  
outpu1=outpaux[:,0]

xx = outpaux[: , 0:1]
yy = outpaux[: , 1:2]

xx = xx.detach().cpu().numpy()
yy = yy.detach().cpu().numpy()
    
tt = t
tp = tt.data.cpu().numpy()
           
plt.figure(figsize = (10,6))
plt.plot(tp,xx,label = "x(t) (Predicted)")
plt.plot(eul_t,eul_x,label = "x(t) (Exact)")
plt.plot(tp,yy,label = "y(t) (Predicted)") 
plt.plot(eul_t,eul_y,label = "y(t) (Exact)") 
plt.xlabel('t')
plt.legend(loc = 'best', fontsize = 19)
plt.title("Epoch Final")
plt.grid()
plt.savefig("LVNDFinal",bbox_inches = 'tight')
plt.show()