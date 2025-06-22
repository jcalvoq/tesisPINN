import torch
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22
   
#Definir la solucion exacta 
def exactSolution(t):
    return np.exp(-t/5)*np.sin(t)

#Funcion para graficar la solucion hasta cierto numero de epochs    
def plotSolution(num_epoch,N,tt,optimizer):
    with torch.no_grad():
        yy = N.forward(tt)
        
    yy = yy.cpu().numpy()
    tt = tt.cpu().numpy()
    
    plt.figure(figsize = (10,6))
    plt.plot(tt,yy,label = "Predicted")
    plt.plot(tt,exactSolution(tt),label = "Exact") 
    plt.title("Epoch " + str(num_epoch))
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig("D:/Tesis/" + optimizer +"/Solution" + str(num_epoch) + ".png",format = "png",bbox_inches = 'tight')
    plt.close()   
    
#Funcion para graficar el error y el valor de loss hasta cierto numero de epochs    
def plotError(num_epoch,vec_loss,vec_error,optimizer):
    ee = np.linspace(0 , num_epoch , num_epoch)
    
    v_loss = np.array(vec_loss)
    v_error = np.array(vec_error)
    
    plt.figure(figsize = (10,6))
    plt.plot(ee,v_error,label = "Exact error")
    plt.plot(ee,v_loss,label = "Loss function")
    plt.xlabel('Epoch') 
    plt.ylabel('Mean square error (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig("D:/Tesis/" + optimizer +"/Error" + str(num_epoch) + ".png",format = "png",bbox_inches = 'tight')
    plt.close()

#Funcion
def plotFinalSolution(tt,uu,optimizer):    
    plt.figure(figsize = (10,6))
    plt.plot(tt,uu,label = "Predicted")
    plt.plot(tt,exactSolution(tt),label = "Exact") 
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(loc = "best")
    plt.title("Epoch 10000")
    plt.grid()
    plt.savefig("D:/Tesis/"+ optimizer + "/FinalSolution.png",format = "png",bbox_inches = 'tight')
    plt.close()

#Funcion
def plotFinalError(e,vec_loss,vec_error,optimizer):
    vec_loss = np.array(vec_loss)
    vec_error = np.array(vec_error)
    ee = np.linspace(0 , e , e)
    plt.figure(figsize = (20,6))
    plt.plot(ee,vec_loss,label = "Loss function" , linewidth = 2)
    plt.plot(ee,vec_error,label = "Exact error" , linewidth = 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean square error (MSE)')
    plt.yscale('log')
    plt.legend(loc = "best")
    plt.title("Loss and Error Values")
    plt.grid()
    plt.savefig("D:/Tesis/" + optimizer + "/FinalError.png",format = "png",bbox_inches = 'tight')
    plt.close()