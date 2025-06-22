import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22

vec_lossR = []
vec_lossF = []
'''
with open('D:\Tesis\ejemplos\Logistic Delay bueno (0,20)\loss.csv' , 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    for line in csv_reader:
        vec_loss.append(line[0])

vec_loss = np.array(vec_loss)  
print(vec_loss)    
ee = np.linspace(0 , 140000 , 140000)

plt.figure(figsize = (20,6))
plt.plot(ee,vec_loss)
plt.xlabel('Epoch')
plt.ylabel('Mean square error (MSE)')
plt.yscale('log')
plt.title("Loss/Error Values")
plt.grid()
plt.show()
'''

#file = pd.read_csv('D:\Tesis\ejemplos\Logistic Delay bueno (0,20)\loss.csv')

f = open('lossR.txt')
g = open('lossF.txt')

for line in f:
    lossR = line.split(' ')[0].strip()
    vec_lossR.append(lossR)

for line in g:
    lossF = line.split(' ')[0].strip()
    vec_lossF.append(lossF)

outputR = []
outputF = []

outputR = [float(string) for string in vec_lossR]
outputF = [float(string) for string in vec_lossF]


#for element in vec_loss:
#    converted = float(element)
#    output.append(converted)

vec_lossR = np.array(vec_lossR)
vec_lossF = np.array(vec_lossF)

e = np.arange(0, 80,1)
print(e)

label=[]
for i in e:
    if i % 50 == 0:
        label.append(i*500)

outputR = np.array(outputR)
outputF = np.array(outputF)

plt.figure(figsize = (20,6))
plt.plot(e,outputR,label = "Loss Function (Rabbits)")
plt.plot(e,outputF,label = "Loss Function (Foxes)")
plt.xlabel('Epoch')
plt.ylabel('Mean square error (MSE)')
plt.yscale('log')

def scale_ticks(x , pos):
    return int(x * 500)
plt.gca().xaxis.set_major_formatter(FuncFormatter(scale_ticks))
plt.title("Loss Values")
plt.grid()
plt.legend(loc = "best")
plt.savefig("LVDFinalError.png",format = "png",bbox_inches = 'tight')
plt.show()