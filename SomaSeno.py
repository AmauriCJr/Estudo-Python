import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0,4*np.pi,0.1)  
y = np.sin(x)

t = np.sin(x + np.pi)

c = np.cos(x + 90)

soma = y + t

plt.plot(x,y,x,t,x,soma)
plt.grid()
plt.show()
