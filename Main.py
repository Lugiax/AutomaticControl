# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:13:00 2015

@author: carlosaranda
Genera datos para el entrenamiento de la red neuronal
"""
from __future__ import division, print_function

from columna2 import reboiler
from AGmultivar import AG
import numpy as np
import time
from Subrutinas import Perturbaciones
from functools import partial
from multiprocessing.pool import Pool

    
reb=reboiler()
dt=0.01

AGS=AG()
AGnInd=30;AGnGen=200
AGS.parametros(optim=0,Nind=AGnInd,Ngen=AGnGen)
AGS.variables(comun=[4,0,10])## 4 variables para los controladores, tomando valores
                            ## mínimos de 0 y máximos de 10

Lvar0=np.zeros(int(1/dt))+20
numero_de_perturbaciones=20
set_de_perturbaciones=Perturbaciones((0,1),dt=dt,n_perts=numero_de_perturbaciones,plot=0)
archivo='resultadosAG4.txt'

def ejecutar(perturbacion):
    Lvar=set_de_perturbaciones[prueba]+Lvar0
    t1=time.time()
    print('--------------------------------\nPerturbación:',prueba+1)
    
    AGS.Fobj(reb.simular,Lvar)
    print('Inicio correcto del AG')
    resultados=AGS.start()
    #reb.simular(resultados[0],Lvar,plot=1)
    ##Escritura de datos en archivo
    with open(archivo,'a') as f:
        res0str=[str(x) for x in Lvar]
        res1str=[str(x) for x in set_de_perturbaciones[prueba]]
        res2str=[str(x) for x in resultados[0]]
        w='<pert>'+','.join(res0str)+'</pert> <pert0>'+','.join(res1str)+'</pert0> <par>'+','.join(res2str)+'</par>\n'
        f.write(w)
        print('Resultados almacenados')
    t2=time.time()
    print('Tiempo por prueba:',(t2-t1)/60)


    

    
    
