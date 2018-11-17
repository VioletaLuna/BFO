#Este archivo usa ecoding: utf-8
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("After630/Data/BFO630_AfterEstaSi_Recal600C_300K_91118_SETd15 #1.txt", delimiter = "", skip_header = 12)
field= data[:,0]
momentum = data[:,1]

plt.plot(field,momentum)
plt.show()
