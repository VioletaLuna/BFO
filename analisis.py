#Este archivo usa ecoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Masa en mg
m630=28.8
nomMedicion = 'After630/Data/BFO630_AfterEstaSi_Recal600C_300K_91118_SETd15 #'
extencion='.txt'
ciclos = np.arange(1,16)
coercitividadIzq = np.zeros(15)
coercitividadDer = np.zeros(15)
ancho= np.zeros(15)
pendiente= np.zeros(15)
magRemanente =np.zeros(15)
magSaturacion = np.zeros(15)


#Hallamos la pendiente
def calPendiente(field, momentum):
	reg= LinearRegression()
	reg = reg.fit(field.reshape(-1, 1),momentum.reshape(-1, 1))
	pendiente=reg.coef_[0][0]
	return pendiente

#Hallamos la coercitividad
def calCoercitividad(field, momentum):
	momentumTem = np.abs(momentum)
	pos1 = np.argmin(momentumTem)
	momentumTem[pos1] = np.inf
	pos2 = np.argmin(momentumTem)
	coerIzq = np.min(np.array([field[pos1], field[pos2]]))
	coerDer = np.max(np.array([field[pos1], field[pos2]]))
	ancho = np.abs(coerIzq - coerDer)
	return coerIzq, coerDer, ancho

#Hallamos la magnetización remanente.
def calMagRemanente(field, momentum):
	fieldTem = np.abs(field)
	pos1 = np.argmin(fieldTem)
	fieldTem[pos1] = np.inf
	pos2 = np.argmin(fieldTem)
	fieldTem[pos2] = np.inf
	pos3 = np.argmin(fieldTem)
	magRema = np.max(np.array([momentum[pos1], momentum[pos2], momentum[pos3]]))
	return magRema

#Hallaos la magnetización de saturación.
def calMagSaturacion(field, momentum):
	pos = np.argmax(field)
	return(momentum[pos])
	
plt.figure(figsize=(10, 7))
for i in ciclos:
	nombre = nomMedicion + str(i) + extencion
	data = np.genfromtxt(nombre, delimiter = "", skip_header = 12)
	fieldTem= data[:,0]
	momentumTem = data[:,1]/m630 #Es necesario normalizar la magnetización con la masa de la muestra 
	plt.plot(fieldTem,momentumTem, label = str(i))
	pendiente[i-1]= calPendiente(fieldTem, momentumTem)
	coercitividadIzq[i-1], coercitividadDer[i-1], ancho[i-1] = calCoercitividad(fieldTem, momentumTem)
	magRemanente[i-1] =calMagRemanente(fieldTem, momentumTem)
	magSaturacion[i-1] = calMagSaturacion(fieldTem, momentumTem)
	
	

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('H(oe)')
plt.ylabel(u'M(emu)')
#plt.ylim(-0.00001,0.00001)
#plt.xlim(-1000,1000)
plt.savefig("Plots/630MvsH15.png" )
plt.close()

plt.plot(ciclos, pendiente, c = "red", label ="Después")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Pendiente')
plt.savefig("Plots/630penvscic.png")
plt.close()

plt.plot(ciclos, coercitividadDer, c = "green", label ="Después")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Coercitividad (oe)')
plt.savefig("Plots/630coervscic.png")
plt.close()

plt.figure(figsize=(11, 7))
plt.plot(ciclos, magRemanente, c = "blue", label ="Después")
plt.xlabel(u'Número de ciclos')
plt.ylabel(u'Magnetización remanente (emu) ')
plt.savefig("Plots/630remavscic.png")
plt.close()

plt.figure(figsize=(9, 7))
plt.plot(ciclos, magSaturacion, c = "violet", label ="Después")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Magnetización de saturación (emu)')
plt.savefig("Plots/630satuvscic.png")
plt.close()

