# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:13:00 2015

@author: carlosaranda
Genera datos para el entrenamiento de la red neuronal
"""
from __future__ import division

from ModeloReboiler import Reboiler
from AGmultivar import AG
import numpy as np
import time
from Subrutinas import Perturbaciones
import matplotlib.pyplot as plt
    
dt=0.01 
numero_de_perturbaciones=100
Lvar0=np.random.random(int(1/dt))*2-1+20
set_de_perturbaciones=Perturbaciones((0,1),dt=dt,
                                     n_perts=numero_de_perturbaciones,
                                     plot=0,sin_pert=True)
print 'Generación de perturbaciones exitosa'
interx=10 #magnitud de las perturbaciones   
intery=3 #valor máximo que pueden tomar los valores de los controladores

AGS=AG(deb=True)
AGnInd=30;AGnGen=100; cores=6
AGS.parametros(optim=0,Nind=AGnInd,Ngen=AGnGen,cores=cores)
AGS.variables(comun=[3,0,intery])## 3 variables para los controladores, tomando 
                            ## valores mínimos de 0 y máximos de 5

#tabla_resultados=list()

archivo='resultadosAG.txt' ##Nombre del archivo en donde se almacenarán los datos
with open(archivo,'w') as f:
    f.write('<prueba>\n<parametros> <nind>{}</nind> <ngen>{}</ngen> \
            <interx>{}</interx> <intery>{}</intery> <dt>{}</dt> </parametros> \
            Ahora se suma tambien los parametros a la funcion a minimizar\n'\
            .format(AGnInd,AGnGen,interx,intery,dt))

print 'Inicio de pruebas'
for prueba in range(numero_de_perturbaciones):
    perturbacion=set_de_perturbaciones[prueba]
    t1=time.time()
    print '--------------------------------\nPerturbación:',prueba+1
    Lvar=perturbacion+Lvar0
#    plt.plot(Lvar)
#    plt.show()
    AGS.Fobj(Reboiler,Lvar)
    print 'Inicio correcto del AG'
    resultados=AGS.start()
    t2=time.time()
    #Reboiler(resultados[0],Lvar,plot=1)
    with open(archivo,'a') as f:
        res0str=[str(x) for x in Lvar]
        res1str=[str(x) for x in perturbacion]
        res2str=[str(x) for x in resultados[0]]
        w='<pert>'+','.join(res0str)+'</pert> <pert0>'+','.join(res1str)+\
            '</pert0> <par>'+','.join(res2str)+'</par>\n'
        f.write(w)
    #tabla_resultados.append([Lvar,set_de_perturbaciones[prueba],resultados[0]])    
    
    print 'Tiempo por prueba:',(t2-t1)/60,'  error:',resultados[1]

with open(archivo,'a') as f:
    f.write('</prueba>')

