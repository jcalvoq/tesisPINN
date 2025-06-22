import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 22

vec_loss = []
vec_error = []
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

f = open('loss.txt')
er = open('error.txt')

for line in f:
    loss = line.split(' ')[0].strip()
    vec_loss.append(loss)
    
for line in er:
    error = line.split(' ')[0].strip()
    vec_error.append(error)

output = []

output = [float(string) for string in vec_loss]
error = [float(string) for string in vec_error]

#for element in vec_loss:
#    converted = float(element)
#    output.append(converted)

vec_loss = np.array(vec_loss)

e = np.arange(0, 400,1)
print(e)

label=[]
for i in e:
    if i % 50 == 0:
        label.append(i*500)

output = np.array(output)

plt.figure(figsize = (20,6))
plt.plot(e,output,label = "Loss Function")
plt.plot(e,error,label = "Exact Error")
plt.plot()
plt.xlabel('Epoch')
plt.ylabel('Mean square error (MSE)')
plt.yscale('log')

def scale_ticks(x , pos):
    return int(x * 500)
plt.gca().xaxis.set_major_formatter(FuncFormatter(scale_ticks))
plt.title("Loss and Error Values")
plt.grid()
plt.legend(loc = "best")
plt.savefig("LogisticFinalError.png",format = "png",bbox_inches = 'tight')
plt.show()

'''
plt.xlabel('Epoch')
plt.ylabel('Mean square error (MSE)')
plt.yscale('log')
plt.xticks(label, rotation=45)
plt.legend(loc = "best")
plt.title("Loss Values")
plt.grid()
plt.show()
'''