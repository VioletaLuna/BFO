#Este archivo usa ecoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Masa en mg
m=np.array([50.7,50.2,43.2])*0.001

tem=np.array(['425','475','630'])
temN=np.array([425,475,630])
nomMedicion1= 'Data/BFO_'
nomMedicion2 ='_SV_300K_MvsH_011018.txt'

campo=np.zeros((396,3))
magnetizacion=np.zeros((396,3))

coercitividadIzq = np.zeros(3)
coercitividadDer = np.zeros(3)
ancho= np.zeros(3)
pendiente= np.zeros(3)
magRemanente =np.zeros(3)
magSaturacion = np.zeros(3)

#Hallamos la pendiente
def calPendiente(field, momentum):
	reg= LinearRegression()
	reg = reg.fit(field.reshape(-1, 1),momentum.reshape(-1, 1))
	pendiente=reg.coef_[0][0]
	return pendiente

#Hallamos la coercitividad
def calCoercitividad(field, momentum):
	#Quitamos los primeros datos para no considerar la liena inicial
	momentumTem = np.abs(momentum[20:])
	pos1 = np.argmin(momentumTem)
	momentumTem[pos1] = np.inf
	pos2 = np.argmin(momentumTem)
	coerIzq = np.min(np.array([field[pos1], field[pos2]]))
	coerDer = np.max(np.array([field[pos1], field[pos2]]))
	ancho = np.abs(coerIzq - coerDer)
	return coerIzq, coerDer, ancho

#Hallamos la magnetización remanente.
def calMagRemanente(field, momentum):
	fieldTem = np.abs(field[20:])
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

for i in range(3):
	nombre = nomMedicion1 + tem[i] + nomMedicion2 
	data = np.genfromtxt(nombre, delimiter = "", skip_header = 12)
	fieldT= data[:,0]
	momentumT = data[:,1]/m[i] #Es necesario normalizar la magnetización con la masa de la muestra
	campo[:,i]=fieldT
	magnetizacion[:,i] = momentumT	 

	plt.plot(fieldT,momentumT, label = tem[i])	
	pendiente[i]= calPendiente(fieldT,momentumT)
	coercitividadIzq[i], coercitividadDer[i], ancho[i] = calCoercitividad(fieldT, momentumT)
	magRemanente[i] =calMagRemanente(fieldT, momentumT)
	magSaturacion[i] = calMagSaturacion(fieldT, momentumT)

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('H(oe)')
plt.ylabel('M(emu/g)')
plt.legend(loc="upper left")
#plt.ylim(-0.000001,0.000001)
#plt.xlim(-100,100)
plt.savefig("Plots/MvsH.png" )
plt.close()

	
plt.figure(figsize=(9,6.5))
plt.axhline(0, color='black')
plt.axvline(0, color='black')
for i in range(3):
	plt.plot(campo[20:,i], magnetizacion[20:,i], label = tem[i])
plt.xlabel('H(oe)')
plt.ylabel('M(emu/g)')
plt.legend(loc="upper left")
plt.ylim(-0.0015,0.0015)
plt.xlim(-150,150)
plt.savefig("Plots/ZoomMvsH.png" )
plt.close()

f, axes = plt.subplots(2, 2, sharex=True,figsize=(18,10) )

axes[0, 0].scatter(temN, coercitividadDer, c='black')
axes[0, 0].set_title('Coercitividad')
axes[0,0].set_ylabel('H (oe)')
axes[0, 0].axvline(temN[0], color='blue')
axes[0, 0].axvline(temN[1], color='green')
axes[0, 0].axvline(temN[2], color='red')

axes[0,1].set_ylim(0,0.000018)
axes[0, 1].scatter(temN, pendiente, c='black')
axes[0, 1].set_title('Pendiente ')
axes[0,1].set_ylabel('(emu/oe g)')
axes[0, 1].axvline(temN[0], color='blue')
axes[0, 1].axvline(temN[1], color='green')
axes[0, 1].axvline(temN[2], color='red')

axes[1, 0].scatter(temN, magRemanente)
axes[1, 0].set_title('Remanente')
axes[1, 0].set_ylabel('M(emu/g)')
axes[1, 0].axvline(temN[0], color='blue')
axes[1, 0].axvline(temN[1], color='green')
axes[1, 0].axvline(temN[2], color='red')

axes[1, 1].scatter(temN, magSaturacion)
axes[1, 1].set_title('Saturación')
axes[1, 1].set_ylabel('M(emu/g)')
axes[1, 1].axvline(temN[0], color='blue')
axes[1, 1].axvline(temN[1], color='green')
axes[1, 1].axvline(temN[2], color='red')

plt.savefig("Plots/valores.png")


