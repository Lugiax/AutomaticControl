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
    
dt=0.01 
numero_de_perturbaciones=20
Lvar0=np.random.random(int(1/dt))+20
set_de_perturbaciones=Perturbaciones((0,1),dt=dt,n_perts=numero_de_perturbaciones,plot=0)
interx=10 #intervalo máximo y mínimo de perturbaciones   
intery=5

AGS=AG()
AGnInd=2;AGnGen=10
AGS.parametros(optim=0,Nind=AGnInd,Ngen=AGnGen)
AGS.variables(comun=[4,0,intery])## 4 variables para los controladores, tomando valores
                            ## mínimos de 0 y máximos de 5

reb=reboiler()
tabla_resultados=list()

with open('resultadosAG.txt','w') as f:
    f.write('<prueba>\n<parametros> <nind>{}</nind> <ngen>{}</ngen> <interx>{}</interx> <intery>{}</intery> <dt>{}</dt> </parametros>\n'.format(AGnInd,AGnGen,interx,intery,dt))
for prueba in range(numero_de_perturbaciones):
    perturbacion=set_de_perturbaciones[prueba]
    t1=time.time()
    print('--------------------------------\nPerturbación:',prueba+1)
    Lvar=perturbacion+Lvar0
    AGS.Fobj(reb.simular,Lvar)
    print('Inicio correcto del AG')
    resultados=AGS.start()
    with open('resultadosAG.txt','a') as f:
        res0str=[str(x) for x in Lvar]
        res1str=[str(x) for x in perturbacion]
        res2str=[str(x) for x in resultados[0]]
        w='<pert>'+','.join(res0str)+'</pert> <pert0>'+','.join(res1str)+'</pert0> <par>'+','.join(res2str)+'</par>\n'
        f.write(w)
    tabla_resultados.append([Lvar,set_de_perturbaciones[prueba],resultados[0]])    
    t2=time.time()
    print('Fin\nTiempo por prueba:',(t2-t1)/60)

with open('resultadosAG.txt','a') as f:
    f.write('</prueba>')

