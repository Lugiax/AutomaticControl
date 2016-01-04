# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:24:45 2016

Entrenamiento de la red neuronal con los resultados de Main.py

@author: carlos
"""
import numpy as np
from RN2 import RedNeuronal
import matplotlib.pyplot as plt
import re

def norm(a,maxmin=None):
    if not maxmin:
        minimo=np.min(a)[0]
        maximo=np.max(a)[0]
    else:
        maximo=maxmin[0];minimo=maxmin[1]
    if not isinstance(a,np.ndarray):
        a=np.array(a)
    return((a-minimo)/(maximo-minimo),(maximo,minimo))
    
    
    
    
    

with open('resultadosAG.txt','r') as f:
    datos=f.read()

npert=int(re.findall('npert:(\d+)',datos)[0])
datosx=re.findall('\n([\d.,]+)',datos)
datosy=re.findall('\[([\d.,]+)\]',datos)
tabla_resultados=list()
for i in range(npert):
    datox=datosx[i].split(',')
    datox=[float(x) for x in datox]
    datoy=datosy[i].split(',')
    datoy=[float(x) for x in datoy]
    tabla_resultados.append([datox,datoy])

dt=0.01

x_ent=list();x_maxmin=list();y_ent=list();y_maxmin=list()
#encontrar maximos y minimos
xmax=-1000;xmin=1000
ymax=-1000;ymin=1000
for res in tabla_resultados:
    xmax1=max(res[0]);xmin1=min(res[0])
    ymax1=max(res[1]);ymin1=min(res[1])
    if xmax1>xmax:
        xmax=xmax1
    if ymax1>ymax:
        ymax=ymax1
    if xmin1<xmin:
        xmin=xmin1
    if ymin1<ymin:
        ymin=ymin1
cont=0
for res in tabla_resultados:
    x,maxmin_x=norm(res[0],(xmax,xmin))
    x=list(x)

    y=list(np.zeros(len(tabla_resultados),dtype=np.int))#Se crea una matriz de zeros del tamaño de la tabla de resultados,
                                     #así la red será entrenada para escojer a un conjunto de elementos
                                     #dependiendo del parecido de los datos de entrada con los de
                                     #entrenamiento
    y[cont]=1
    x_ent.append(list(x));x_maxmin.append(maxmin_x)
    y_ent.append(list(y))
    cont+=1

''' 
Entrenamiento de la red neuronal
'''
pImp=([3,3],30,1000)
tipo_ent='EL'
datos=(np.array(x_ent),np.array(y_ent))
print('\nInicio de la red')
red=RedNeuronal(estructura=pImp[0],deb=True)
pesos=red.Entrenar(datos_ent=datos,tipo_entrenamiento=tipo_ent,parametrosAG=(pImp[1],pImp[2]),pruebasAG=3) 

print('\nFinalización de entrenamiento, se comienza a almacenar los pesos')

with open('./pruebasent1.txt','a') as f:
    info='Datos de corrida:pImp['+str(pImp[0][0])+','+str(pImp[0][1])+','+str(pImp[1])+','+str(pImp[2])+','+str(dt)+'] tipo:'+tipo_ent+'\n'
    f.write(info)
    p=''
    r=''
    for i in pesos:
        for k in i:
            for l in k:
                p=p+str(l)+','
    for j in range(len(tabla_resultados)):
        r=r+'['
        for i in tabla_resultados[j][1]:
            r=r+str(i)+','
        r=r+'],'    
    w=p[:len(p)-1]+'/'+r[:len(r)-1]+'\n' #Todo junto
    f.write(w)
       
print('\nFin de la operación')