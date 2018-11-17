#Este archivo usa ecoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Masa en mg
m630=28.8
data = np.genfromtxt("After630/Data/BFO630_AfterEstaSi_Recal600C_300K_91118_SETd15 #1.txt", delimiter = "", skip_header = 12)
field= data[:,0]
momentum = data[:,1]/m630 #Es necesario normalizar la magnetización con la masa de la muestra 


#Hallamos la pendiente
def pendiente()
	reg= LinearRegression()
	reg = reg.fit(field.reshape(-1, 1),momentum.reshape(-1, 1))
	pendiente=reg.coef_[0][0]
	return pendiente

#Hallamos la coercitividad
def coercitividad():
	momentumTem = np.abs(momentum)
	pos1 = np.argmin(momentumTem)
	momentumTem[pos1] = np.inf
	pos2 = np.argmin(momentumTem)
	coerIzq = np.min(np.array([field[pos1], field[pos2]]))
	coerDer = np.max(np.array([field[pos1], field[pos2]]))
	return coerIzq, coerDer

#Hallamos la magnetización remanente.
def magRemanente():
	fieldTem = np.abs(field)
	pos1 = np.argmin(fieldTem)
	fieldTem[pos1] = np.inf
	pos2 = np.argmin(fieldTem)
	fieldTem[pos2] = np.inf
	pos3 = np.argmin(fieldTem)
	magRema = np.max(np.array([momentum[pos1], momentum[pos2], momentum[pos3]]))
	return magRema

#Hallaos la magnetización de saturación.
def magSaturacion():
	pos = np.argmax(field)
	return(momentum[pos])
	


rema = magRemanente()
satu = magSaturacion()

plt.plot(field,momentum)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('H(oe)')
plt.ylabel(u'M(emu)')
#plt.ylim(-0.00003,0.00003)
#plt.xlim(-1000,1000)


plt.show()
