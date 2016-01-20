# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:45:43 2015

@author: carlosaranda
Prueba final Red Neuronal para control
"""
from __future__ import division
import re
from columna2 import reboiler
from RN2 import RedNeuronal, redimensionarPesos, dimensionarMatricesDePesos
import numpy as np
from Subrutinas import norm, Perturbaciones, denorm
import matplotlib.pyplot as plt


with open('pruebasent.txt','r') as f:
    archivo=f.read()
    

est=re.findall('\<estructura>([-\d.,]+)',archivo)[0]
pesos=re.findall('\<pesos>([-\d.,]+)',archivo)[0]
xmaxmin=float(re.findall('\<interx>([-\d.,]+)',archivo)[0])
ymax=float(re.findall('\<intery>([-\d.,]+)',archivo)[0])
est=[int(z) for z in est.split(',')]
pesos=[float(z) for z in pesos.split(',')]

'''
Se inicializa el reboiler con todos sus parámetros
'''
reb=reboiler()
## Propiedades de las substancias 
reb.alpha=2.54
cpA=2.42;cpB=4.18;reb.cpsubs=(cpA,cpB)
lamvapA=854.;lamvapB=2260.;reb.lamvapsubs=(lamvapA,lamvapB)
tonoA=(8.20417,1642.89,230.3);tonoB=(8.0713,1730.63,233.426);reb.tono=(tonoA,tonoB)
##Control LOS VALORES DE LOS CONTROLADORES SERÁN MODIFICADOS POR EL AG
reb.kcb=.8;reb.tdb=.5;reb.Bref=9 ##Para fondos
reb.kcq=2;reb.tdq=1.5 ##Para reboiler
reb.Mref=30. ##Para la masa del interior del reboiler
dt=0.01;t=0
Ml,Bl,Vl,tl,Tl=None,None,None,None,None

## Condiciones iniciales:
reb.L=20;reb.xl=0.8
reb.M=30.;reb.Q=1.1e4;reb.B=10
## Estado Estacionario: reb.M=30;reb.Q=1.08e4;reb.B=9;reb.V=11
reb.x=reb.xl;reb.y=reb.equil(reb.x);lamvap=reb.lamvap_f(reb.x)
reb.hl=220.;reb.h=reb.hl;reb.H=reb.h+lamvap
reb.T=reb.h/reb.cp(reb.x);reb.V=reb.Q/reb.lamvap_f(reb.x)
reb.Pt=760.##mmHg
t=0
Ml,Bl,Vl,tl,Tl,xli=[reb.M],[reb.B],[reb.V],[t],[reb.T],[reb.x]

## Se crean las perturbaciones
nperts=20
set_de_perturbaciones=Perturbaciones((0,1),dt=dt,n_perts=nperts,tipo=1)

Lvar=list()
base=20
for j in range(len(set_de_perturbaciones)*2):
    perturbacion=set_de_perturbaciones[int(j/2)]
    if j%2==0:
        for k in perturbacion:
            Lvar.append(k+base)
    else:
        if len(Lvar)!=0:
            base=Lvar[-1]
        Lvar0=list((np.random.rand(int(1/dt))*2-1)*.1+base)
        for k in Lvar0:
            Lvar.append(k)

plt.plot(Lvar)
plt.show()

'''
Se inicia la red neuronal
'''
red=RedNeuronal(estructura=est[1:len(est)-1],neurodos_entrada_salida=(int(1/dt),4))

est,nPesos=dimensionarMatricesDePesos(red.estructura)
W=redimensionarPesos(pesos,est)
contr_l=[[0],[0],[0],[0]]
cont=[0,0,0,0]
for i in range(len(Lvar)):
    t+=dt
    reb.L=float(Lvar[i])
    reb.actualizar(t,dt)
    Ml.append(reb.M),Bl.append(reb.B),Vl.append(reb.V),Tl.append(reb.T),tl.append(t),xli.append(reb.x)
    contr_l[0].append(cont[0]);contr_l[1].append(cont[0]);contr_l[2].append(cont[1]);contr_l[3].append(cont[3])
    if i < int(1/dt):
        continue
    else:
        ventana=norm(Lvar[i-int(1/dt):i],(xmaxmin,-xmaxmin))-20
        cont=denorm(red.FP(pesos=W,xi=ventana)[-1],(ymax,0))
        reb.kcb,reb.tdb,reb.kcq,reb.tdq=cont

        if i%100000==0:
            print 'Los valores de los controladores son:{}'.format(cont)
            
            plt.figure(figsize=(16,10))
           
            plt.subplot(2,1,1);plt.grid(True)
            plt.plot(tl,Ml,'b.',label='Acumulacion')
            plt.plot(tl,Bl,'g.',label='Fondos')
            plt.plot(tl,Vl,'r.',label='Vapor')
            plt.xlabel('tiempo');plt.ylabel('kg/min')
            #plt.legend(loc=4)
            
            plt.subplot(2,1,2);plt.grid(True)
            plt.title('Flujo de entrada')
            plt.plot(ventana)
            plt.xlabel('tiempo');plt.ylabel('Flujo de entrada')
            
#            
#            plt.subplot(2,2,2);plt.grid(True)
#            plt.plot(tl,Tl,'b.')
#            plt.xlabel('tiempo');plt.ylabel('Temperatura')
            

            
#            plt.subplot(2,2,4);plt.grid(True)
#            plt.title('FraccionMolar')
#            plt.plot(tl,xli,'b.')
#            plt.xlabel('tiempo');plt.ylabel('X')
            
            plt.show()
    
    
    
    
plt.figure(figsize=(16,10))
   
plt.subplot(2,2,1);plt.grid(True)
plt.plot(tl,Ml,'b.',label='Acumulacion')
plt.plot(tl,Bl,'g.',label='Fondos')
plt.plot(tl,Vl,'r.',label='Vapor')
plt.xlabel('tiempo');plt.ylabel('kg/min')

plt.subplot(2,2,3);plt.grid(True)
plt.title('Flujo de entrada')
plt.plot(np.array(Lvar)-20)
plt.xlabel('tiempo');plt.ylabel('Flujo de entrada')   

plt.subplot(2,2,2);plt.grid(True)
plt.title('Controlador proporcional de reboiler')
plt.plot(tl,contr_l[2],'r.',label='Vapor')

plt.xlabel('tiempo');plt.ylabel('valores de controladores')

plt.subplot(2,2,4);plt.grid(True)
plt.title('Controlador diff de reboiler')
plt.plot(tl,contr_l[3],'y.',label='Vapor')
#plt.plot(tl,contr_l[0],'b.',label='Acumulacion')
#plt.plot(tl,contr_l[1],'g.',label='Fondos')
plt.xlabel('tiempo');plt.ylabel('valores de controladores')

plt.show()