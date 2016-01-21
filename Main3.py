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

for i in range(npert):
    datox=datosx[i].split(',')
    datox=[float(x) for x in datox]
    datoy=datosy[i].split(',')
    datoy=[float(x) for x in datoy]
    if len(datox)==100 and len(datoy)==4:
        tabla_resultados.append([datox,datoy])
    else:
        print 'La perturbacion {} no cumple con las dimensiones-> x:{} ; y:{}'.format(i+1,len(datosx[i]),len(datosy[i]))

x_ent=list();y_ent=list()
for res in tabla_resultados:
    x=list(norm(res[0],(interx,-interx)))
    y=list(norm(res[1],(intery,0)))
    x_ent.append(list(x))
    y_ent.append(list(y))

''' 
Entrenamiento de la red neuronal
'''
tipo_ent='CM'
datos=(np.array(x_ent),np.array(y_ent))
print '\nInicio de la red'
estructura=[15,10,5]
red=RedNeuronal(estructura=estructura,deb=True)
pesos=red.Entrenar(datos_ent=datos,tipo_entrenamiento=tipo_ent) 

print '\nFinalización de entrenamiento, se comienza a almacenar los pesos'

with open('./pruebasent3.txt','w') as f:
    info='<entrenamiento><estructura>{}</estructura><interx>{}</interx><intery>{}</intery><dt>{}</dt>\n'.format(','.join([str(x for x in estructura)]),interx,intery,dt)
    f.write(info)
    p=''
    for i in pesos:
        for k in i:
            for l in k:
                p=p+str(l)+','
   
    w=p[:-1]+'\n' #Todo junto
    f.write('<pesos>'+w+'</pesos>\</entrenamiento>')
       
print '\nFin de la operación'