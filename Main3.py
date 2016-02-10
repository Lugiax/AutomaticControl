# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:24:45 2016

Entrenamiento de la red neuronal con los resultados de Main.py

@author: carlos
"""
from __future__ import division
import numpy as np
from RN2 import RedNeuronal
import matplotlib.pyplot as plt
import re
from Subrutinas import norm    

with open('resultadosAG.txt','r') as f:
    datos=f.read()

datosx=re.findall('\<pert0\>([-\d.,]+)',datos)
datosy=re.findall('\<par\>([-\d.,]+)',datos)
interx=float(re.findall('\<interx\>([-\d.]+)',datos)[0])
intery=float(re.findall('\<intery\>([-\d.]+)',datos)[0])
dt=float(re.findall('\<dt\>([-\d.]+)',datos)[0])

tabla_resultados=list()
npert=len(datosx)
contr_l=[list(),list(),list()]
for i in range(npert):
    datox=datosx[i].split(',')
    datox=[float(x) for x in datox]
    datoy=datosy[i].split(',')
    datoy=[float(x) for x in datoy]
    contr_l[0].append(datoy[0])
    contr_l[1].append(datoy[1])
    contr_l[2].append(datoy[2])
    if len(datox)==100 and len(datoy)==3:
        tabla_resultados.append([datox,datoy])
    else:
        print 'La perturbacion {} no cumple con las dimensiones-> x:{} ; y:{}'.format(i+1,len(datosx),len(datosy))
plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plt.plot(contr_l[0])
plt.title('Controlador Proporcional')
plt.subplot(1,3,2)
plt.plot(contr_l[1])
plt.title('Controlador Derivativo')
plt.subplot(1,3,3)
plt.plot(contr_l[2])
plt.title('Controlador Integrativo')
plt.show()
x_ent=list();y_ent=list()
for res in tabla_resultados:
    x=list(norm(res[0],(interx,-interx)))
    y=list(norm(res[1],(intery,0)))
    x_ent.append(list(x))
    y_ent.append(list(y))

''' 
Entrenamiento de la red neuronal
'''
tipo_ent='EL'
datos=(np.array(x_ent),np.array(y_ent))
print '\nInicio de la red'
estructura=[10,10,5]
red=RedNeuronal(estructura=estructura,deb=True)
pesos=red.Entrenar(datos_ent=datos,tipo_entrenamiento=tipo_ent,max_iter=1000000) 

print '\nFinalización de entrenamiento, se comienza a almacenar los pesos'

with open('./pruebasent.txt','w') as f:
    info='<entrenamiento><estructura>{}</estructura><interx>{}</interx><intery>{}</intery><dt>{}</dt>\n'.format(','.join([str(k) for k in red.estructura]),interx,intery,dt)
    f.write(info)
    p=''
    for i in pesos:
        for k in i:
            for l in k:
                p=p+str(l)+','
   
    w=p[:-1]+'\n' #Todo junto
    f.write('<pesos>'+w+'</pesos>\</entrenamiento>')
       
print '\nFin de la operación'