#Este archivo usa ecoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Masa en mg
m630After=28.8
m630Before =43.2

nomMedicionAfter = 'After630/BFO630_AfterEstaSi_Recal600C_300K_91118_SETd15 #'
nomMedicionBefore ='Before630/BFO630_BeforeEstaSi_Recal600C_300K_91118_SETd15 #'
extencion='.txt'

ciclos = np.arange(1,16)

coercitividadIzqAfter = np.zeros(15)
coercitividadIzqBefore = np.zeros(15)

coercitividadDerAfter = np.zeros(15)
coercitividadDerBefore = np.zeros(15)

anchoAfter= np.zeros(15)
anchoBefore= np.zeros(15)

pendienteAfter= np.zeros(15)
pendienteBefore= np.zeros(15)

magRemanenteAfter =np.zeros(15)
magRemanenteBefore =np.zeros(15)

magSaturacionAfter = np.zeros(15)
magSaturacionBefore = np.zeros(15)




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
	nombreAfter = nomMedicionAfter + str(i) + extencion
	data = np.genfromtxt(nombreAfter, delimiter = "", skip_header = 12)
	fieldAfter= data[:,0]
	momentumAfter = data[:,1]/m630After #Es necesario normalizar la magnetización con la masa de la muestra
	
	nombreBefore = nomMedicionBefore + str(i) + extencion
	data = np.genfromtxt(nombreBefore, delimiter = "", skip_header = 12)
	fieldBefore= data[:,0]
	momentumBefore = data[:,1]/m630Before #Es necesario normalizar la magnetización con la masa de la muestra
	 
	plt.plot(fieldAfter,momentumAfter, label = str(i), c="red")	
	pendienteAfter[i-1]= calPendiente(fieldAfter, momentumAfter)
	coercitividadIzqAfter[i-1], coercitividadDerAfter[i-1], anchoAfter[i-1] = calCoercitividad(fieldAfter, momentumAfter)
	magRemanenteAfter[i-1] =calMagRemanente(fieldAfter, momentumAfter)
	magSaturacionAfter[i-1] = calMagSaturacion(fieldAfter, momentumAfter)
	
	plt.plot(fieldBefore,momentumBefore, label = str(i), c= "green")	
	pendienteBefore[i-1]= calPendiente(fieldBefore, momentumBefore)
	coercitividadIzqBefore[i-1], coercitividadDerBefore[i-1], anchoBefore[i-1] = calCoercitividad(fieldBefore, momentumBefore)
	magRemanenteBefore[i-1] =calMagRemanente(fieldBefore, momentumBefore)
	magSaturacionBefore[i-1] = calMagSaturacion(fieldBefore, momentumBefore)
	

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('H(oe)')
plt.ylabel(u'M(emu)')
#plt.ylim(-0.00001,0.00001)
#plt.xlim(-1000,1000)
plt.savefig("Plots/630MvsH15.png" )
plt.close()

plt.plot(ciclos, pendienteAfter, c = "red", label ="Después")
plt.plot(ciclos, pendienteBefore, c = "green", label ="Antes")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Pendiente')
plt.legend()
plt.savefig("Plots/630penvscic.png")
plt.close()

plt.plot(ciclos, coercitividadDerAfter, c = "red", label ="Después")
plt.plot(ciclos, coercitividadDerBefore, c = "green", label ="Antes")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Coercitividad (oe)')
plt.legend(loc="lower right")
plt.savefig("Plots/630coervscic.png")
plt.close()

plt.figure(figsize=(11, 7))
plt.plot(ciclos, magRemanenteAfter, c = "red", label ="Después")
plt.plot(ciclos, magRemanenteBefore, c = "green", label ="Antes")
plt.xlabel(u'Número de ciclos')
plt.ylabel(u'Magnetización remanente (emu)')
plt.legend()
plt.savefig("Plots/630remavscic.png")
plt.close()

plt.figure(figsize=(9, 7))
plt.plot(ciclos, magSaturacionAfter, c = "red", label ="Después")
plt.plot(ciclos, magSaturacionBefore, c = "green", label ="Antes")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Magnetización de saturación (emu)')
plt.legend()
plt.savefig("Plots/630satuvscic.png")
plt.close()

plt.plot(ciclos, anchoAfter, c = "red", label ="Después")
plt.plot(ciclos, anchoBefore, c = "green", label ="Antes")
plt.xlabel('Número de ciclos')
plt.ylabel(u'Ancho histeresis (oe)')
plt.legend()
plt.savefig("Plots/630anchovscic.png")
plt.close()
