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


def norm(a,maxmin=None):
    if not maxmin:
        minimo=np.min(a)[0]
        maximo=np.max(a)[0]
    else:
        maximo=maxmin[0];minimo=maxmin[1]
    if not isinstance(a,np.ndarray):
        a=np.array(a)
    return((a-minimo)/(maximo-minimo),(maximo,minimo))

def perturbacion(pert=None,Lvar0=None,inter=(0,1)):
    Lvar=np.copy(Lvar0)
    #print('Perturbaciones de:')
    for i in range(1):
        inicio=inter[0]
        intervalo=inter[1]-inter[0]
        fin=int(inicio+intervalo)
        x=0
        #print(inicio,'-',fin,'min') 
        for j in range(int(inicio/dt),int(fin/dt)):
            if pert==0:
                Lvar[j]=Lvar[j]+gauss(x*dt)
            elif pert==1:
                Lvar[j]=Lvar[j]-gauss(x*dt)
            elif pert==2:
                Lvar[j]=Lvar[j]+(x*dt)*3
            elif pert==3:
                Lvar[j]=Lvar[j]-(x*dt)*3
            elif pert==4:
                Lvar[j]=Lvar[j]-(x*dt)*3+gauss(x*dt)
            elif pert==5:
                Lvar[j]=Lvar[j]+(x*dt)*3-gauss(x*dt)
            x+=1
    return(Lvar)
    
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

def f_obj(controladores,op):
    reb.kcb,reb.tdb,reb.kcq,reb.tdq=controladores
    condini()
    for i in range(int(tf/dt)):
        global t
        t+=dt
        reb.L=float(Lvar[i])
        reb.actualizar(t,dt)
        Ml.append(reb.M),Bl.append(reb.B),Vl.append(reb.V),Tl.append(reb.T),tl.append(t),xli.append(reb.x)
    return((Ml[-1]-reb.Mref+Bl[-1]-reb.Bref)**2)
    
##############################################################################################    
AGS=AG()
AGnInd=50;AGnGen=400
AGS.parametros(optim=0,Nind=AGnInd,Ngen=AGnGen)
AGS.variables(comun=[4,0,5])## 4 variables para los controladores, tomando valores
                            ## mínimos de 0 y máximos de 5

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

'''
Para simular un sistema inestable, se modificará el valor de L en base a una función dada
'''
np.random.seed(10)
Lvar0=np.random.random(int(tf/dt))+20
gauss= lambda x:10*np.exp(-(x-0.5)**2/(2*0.05**2))

'''
--------------
'''
n_perturbaciones=6 #6max
n_pruebas=1
'''
---------
'''

tabla_resultados=list()
for prueba in range(n_perturbaciones):
    t1=time.time()
    print('--------------------------------\nPerturbación:',prueba+1)
    Lvar=perturbacion(prueba,Lvar0)
    AGS.Fobj(f_obj)
    resultados_plot=([],1000)
    for i in range(n_pruebas):
        print('Corrida:',i+1)
        print('Inicio correcto del AG')
        resultados=AGS.start()
        if resultados[1]<resultados_plot[1]:
            resultados_plot=resultados
        print('Fin de la corrida:',i+1,'\n------------')
    #print('Fin de la prueba:{}, resultados:{}'.format(prueba,resultados_plot))
    tabla_resultados.append([Lvar,resultados_plot[0]])
#    plt.plot(tabla_resultados[prueba][0])
#    plt.show()    
    t2=time.time()
    print('Tiempo por prueba:',(t2-t1)/60)

## Se escriben los resultados recien obtenidos
with open('resultadosAG2.txt','a') as f:
    f.write('resultadosAG npert:{} - Nind:{} - Ngen:{}\n'.format(n_perturbaciones,AGnInd,AGnGen))
    for res in tabla_resultados:
        res0str=[str(x) for x in res[0]]
        res1str=[str(x) for x in res[1]]
        w=','.join(res0str)+'/['+','.join(res1str)+']\n'
        f.write(w)
    
    print('Resultados almacenados\n')
