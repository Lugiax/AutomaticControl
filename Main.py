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
    
def condini():
    ## Condiciones iniciales:
    reb.L=20;reb.xl=0.8
    reb.M=30.;reb.Q=1.1e4;reb.B=9
    ## Estado Estacionario: reb.M=30;reb.Q=1.08e4;reb.B=9;reb.V=11
    reb.x=reb.xl;reb.y=reb.equil(reb.x);lamvap=reb.lamvap_f(reb.x)
    reb.hl=220.;reb.h=reb.hl;reb.H=reb.h+lamvap
    reb.T=reb.h/reb.cp(reb.x);reb.V=reb.Q/reb.lamvap_f(reb.x)
    reb.Pt=760.##mmHg
    global t
    t=0
    global Ml,Bl,Vl,tl,Tl,xli
    Ml,Bl,Vl,tl,Tl,xli=[reb.M],[reb.B],[reb.V],[t],[reb.T],[reb.x]

def f_obj(controladores,plot=0):
    reb.kcb,reb.tdb,reb.kcq,reb.tdq=controladores
    condini()
    for i in range(int(tf/dt)):
        global t
        t+=dt
        reb.L=float(Lvar[i])
        reb.actualizar(t,dt)
        Ml.append(reb.M),Bl.append(reb.B),Vl.append(reb.V),Tl.append(reb.T),tl.append(t),xli.append(reb.x)
    if plot==1:
        print('Graficando la simulacion')
        plt.figure(figsize=(16,10))
       
        plt.subplot(2,2,1);plt.grid(True)
        plt.plot(tl,Ml,'b.',label='Acumulacion')
        plt.plot(tl,Bl,'g.',label='Fondos')
        plt.plot(tl,Vl,'r.',label='Vapor')
        plt.xlabel('tiempo');plt.ylabel('kg/min')
        plt.legend(loc=4)
        
        plt.subplot(2,2,2);plt.grid(True)
        plt.plot(tl,Tl,'b.')
        plt.xlabel('tiempo');plt.ylabel('Temperatura')
        
        plt.subplot(2,2,3);plt.grid(True)
        plt.title('Perturbacion')
        plt.plot(Lvar)
        plt.xlabel('tiempo');plt.ylabel('Flujo de entrada')
        
        plt.subplot(2,2,4);plt.grid(True)
        plt.title('FraccionMolar')
        plt.plot(tl,xli,'b.')
        plt.xlabel('tiempo');plt.ylabel('X')
        
        plt.show()
    return((Ml[-1]-reb.Mref+Bl[-1]-reb.Bref)**2)
    
##############################################################################################  
    
reb=reboiler()
## Propiedades de las substancias 
reb.alpha=2.54
cpA=2.42;cpB=4.18;reb.cpsubs=(cpA,cpB)
lamvapA=854.;lamvapB=2260.;reb.lamvapsubs=(lamvapA,lamvapB)
tonoA=(8.20417,1642.89,230.3);tonoB=(8.0713,1730.63,233.426);reb.tono=(tonoA,tonoB)
##Control LOS VALORES DE LOS CONTROLADORES SERÁN MODIFICADOS POR EL AG
reb.kcb=0;reb.tdb=0;reb.Bref=9 ##Para fondos
reb.kcq=0;reb.tdq=0 ##Para reboiler
reb.Mref=30. ##Para la masa del interior del reboiler
dt=0.01;tf=1;t=0
Ml,Bl,Vl,tl,Tl,xli=None,None,None,None,None,None


AGS=AG()
AGnInd=50;AGnGen=400
AGS.parametros(optim=0,Nind=AGnInd,Ngen=AGnGen)
AGS.variables(comun=[4,0,6])## 4 variables para los controladores, tomando valores
                            ## mínimos de 0 y máximos de 5

Lvar0=np.random.random(int(tf/dt))+20
numero_de_perturbaciones=2
set_de_perturbaciones=Perturbaciones((0,1),dt=dt,n_perts=numero_de_perturbaciones,plot=0)

tabla_resultados=list()
for prueba in range(numero_de_perturbaciones):
    t1=time.time()
    print('--------------------------------\nPerturbación:',prueba+1)
    Lvar=set_de_perturbaciones[prueba]
    AGS.Fobj(f_obj)
    print('Inicio correcto del AG')
    resultados=AGS.start()
    #print('Fin de la prueba:{}, resultados:{}'.format(prueba,resultados_plot))
    tabla_resultados.append([Lvar,resultados[0]])
#    plt.plot(tabla_resultados[prueba][0])
#    plt.show()    
    t2=time.time()
    print('Fin\nTiempo por prueba:',(t2-t1)/60)

## Se escriben los resultados recien obtenidos
with open('resultadosAG.txt','w') as f:
    f.write('resultadosAG | Parametros-> Nind:{} - Ngen:{}\n'.format(AGnInd,AGnGen))
    for res in tabla_resultados:
        res0str=[str(x) for x in res[0]]
        res1str=[str(x) for x in res[1]]
        w='<pert>'+','.join(res0str)+'</pert><par>'+','.join(res1str)+'</par>\n'
        f.write(w)
    
    print('Resultados almacenados\n')
