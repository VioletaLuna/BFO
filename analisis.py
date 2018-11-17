#Este archivo usa ecoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Masa en mg
m630=28.8
data = np.genfromtxt("After630/Data/BFO630_AfterEstaSi_Recal600C_300K_91118_SETd15 #1.txt", delimiter = "", skip_header = 12)
field= data[:,0]
#Es necesario normalizar la magnetización con la masa de la muestra 
momentum = data[:,1]/m630

reg= LinearRegression()
reg = reg.fit(field.reshape(-1, 1),momentum.reshape(-1, 1))
pendiente=reg.coef_[0][0]
print(pendiente)

plt.plot(field,momentum)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('H(oe)')
plt.ylabel(u'M(emu)')


#plt.show()
