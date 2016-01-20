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
from Subrutinas import norm
    
with open('resultadosAG4.txt','r') as f:
    datos=f.read()


datosx=re.findall('\<pert0\>([-\d.,]+)',datos)##Selecciona los datos para entrenar a la red
datosy=re.findall('\<par\>([-\d.,]+)',datos)
tabla_resultados=list()
npert=len(datosx)
for i in range(npert):
    datox=datosx[i].split(',')
    datox=[float(x) for x in datox]
    datoy=datosy[i].split(',')
    datoy=[float(x) for x in datoy]
    if len(datox)!=100 or len(datoy)!=4:
        print 'Se descarta perturbacion {}, No datosx:{} ; No datosy:{}'.format(i+1,len(datox),len(datosy))
        continue
    tabla_resultados.append([datox,datoy])

print 'Numero de resultados correctos',len(tabla_resultados)

dt=0.01

x_ent=list();x_maxmin=list();y_ent=list();y_maxmin=list()

xmaxmin=10
ymax=10
cont=1
for res in tabla_resultados:
    x=norm(res[0],(xmaxmin,-xmaxmin))
#    plt.plot(x)
#    plt.title('Perturbación'+str(cont))
#    plt.show()
    cont+=1
    
    y=norm(res[1],(ymax,0))
    x_ent.append(list(x))
    y_ent.append(list(y))


''' 
Entrenamiento de la red neuronal
'''
est=[20,10,10]
tipo_ent='EL'
datos=(np.array(x_ent),np.array(y_ent))
print('\nInicio de la red')
red=RedNeuronal(estructura=est,deb=True)
pesos=red.Entrenar(datos_ent=datos,max_iter=1000000) 

print('\nFinalización de entrenamiento, se comienza a almacenar los pesos')

with open('./pruebasent1.txt','w') as f:
    est_str=[str(z) for z in red.estructura]
    info='<prueba>\n<estructura>{}</estructura><interx>{}</interx><intery>{}</intery>\n'.format(','.join(est_str),xmaxmin,ymax)
    f.write(info)
    p=''
    r=''
    for i in pesos:
        for k in i:
            for l in k:
                p=p+str(l)+','  
    w='<pesos>'+p[:len(p)-1]+'</pesos>\n</prueba>\n'#+r[:len(r)-1]+'\n' #Todo junto
    f.write(w)
       
print('\nFin de la operación')