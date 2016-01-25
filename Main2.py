# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:45:43 2015

@author: carlosaranda
Prueba final Red Neuronal para control
"""
import re
from columna2 import reboiler
from RN2 import RedNeuronal, redimensionarPesos, dimensionarMatricesDePesos
import numpy as np
from Subrutinas import Perturbaciones, denorm, norm
import matplotlib.pyplot as plt


with open('pruebasent.txt','r') as f:
    archivo=f.read()
    

cont=-1
pesos=re.findall('\<pesos\>([-\d.,]+)',archivo)[0]
estructura=re.findall('\<estructura\>([\d,]+)',archivo)[0]
interx=float(re.findall('\<interx\>([-\d.]+)',archivo)[0])
intery=float(re.findall('\<intery\>([-\d.]+)',archivo)[0])
dt=float(re.findall('\<dt\>([-\d.]+)',archivo)[0])

pesos=[float(x) for x in pesos.split(',')]
estructura=[int(x) for x in estructura.split(',')]
'''
Se inicializa el reboiler con todos sus parámetros
'''
n_perts=20
set_de_perturbaciones=Perturbaciones((0,1),n_perts=n_perts)
Lvar=[]
base=20
for i in range(2*n_perts):
    if i%1000==3:
        for k in set_de_perturbaciones[int(i/2)]:
            Lvar.append(k+base)
    else:
        Lvar0=(np.random.rand(int(1/dt))*2-1)*.1+base
        for k in Lvar0:
            Lvar.append(k)
    base=Lvar[-1]
    
plt.plot(Lvar)
plt.show()
#se añaden las perturbaciones

'''
Se inicia la red neuronal
'''
ar=estructura[1:-1]
neurodos_entrada_salida=(estructura[0],estructura[-1])
red=RedNeuronal(estructura=ar,neurodos_entrada_salida=neurodos_entrada_salida)

est,nPesos=dimensionarMatricesDePesos(red.estructura)
W=redimensionarPesos(pesos,est)

reb=reboiler()
reb.condini()
t=0
Ml,Bl,Vl,tl,Tl,xli,kc,td=[reb.M],[reb.B],[reb.V],[t],[reb.T],[reb.x],[reb.kcq],[reb.tdq]
for i in range(len(Lvar)):
    t+=dt
    reb.L=Lvar[i]
    reb.actualizar(t,dt)
    Ml.append(reb.M),Bl.append(reb.B),Vl.append(reb.V),Tl.append(reb.T)
    tl.append(t),xli.append(reb.x),kc.append(reb.kcq),td.append(reb.tdq)
    if i < int(1/dt):
        continue
    else:
        ventana=[Lvar[i-int(1/dt):i]]
        ventana_norm=norm(ventana,(interx+20,20-interx))
        y=red.FP(pesos=W,xi=ventana)[-1]
        reb.kcb,reb.tdb,reb.kcq,reb.tdq=denorm(y,(intery,0))
        
        if i%1000==0:
            plt.figure(figsize=(16,10))
               
            plt.subplot(2,2,1);plt.grid(True)
            plt.plot(tl[i-int(1/dt):i],Ml[i-int(1/dt):i],'b.',label='Acumulacion')
            plt.plot(tl[i-int(1/dt):i],Bl[i-int(1/dt):i],'g.',label='Fondos')
            plt.plot(tl[i-int(1/dt):i],Vl[i-int(1/dt):i],'r.',label='Vapor')
            plt.xlabel('tiempo');plt.ylabel('kg/min')
            #plt.legend(loc=4)
            
            plt.subplot(2,2,2);plt.grid(True)
            plt.title('Control Proporcional')
            plt.plot(tl[i-int(1/dt):i],kc[i-int(1/dt):i],'b.')
            plt.xlabel('tiempo');plt.ylabel('Var Prop')
            
            plt.subplot(2,2,3);plt.grid(True)
            plt.title('Perturbacion')
            plt.plot(tl[i-int(1/dt):i],ventana_norm[0])#Lvar[i-int(1/dt):i])
            plt.xlabel('tiempo');plt.ylabel('Flujo de entrada')
            
            plt.subplot(2,2,4);plt.grid(True)
            plt.title('Control Derivativo')
            plt.plot(tl[i-int(1/dt):i],td[i-int(1/dt):i],'b.')
            plt.xlabel('tiempo');plt.ylabel('Var Derivativa')
            
            plt.show()

print('Graficando la simulacion')
plt.figure(figsize=(16,10))
   
plt.subplot(2,2,1);plt.grid(True)
plt.xlim((0,2*n_perts))
plt.plot(tl,Ml,'b.',label='Acumulacion')
plt.plot(tl,Bl,'g.',label='Fondos')
plt.plot(tl,Vl,'r.',label='Vapor')
plt.xlabel('tiempo');plt.ylabel('kg/min')
#plt.legend(loc=4)

plt.subplot(2,2,2);plt.grid(True)
plt.xlim((0,2*n_perts))
plt.title('Control Proporcional')
plt.plot(tl,kc,'b.')
plt.xlabel('tiempo');plt.ylabel('control proporcional')

plt.subplot(2,2,3);plt.grid(True)
plt.title('Perturbacion')
plt.plot(norm(Lvar,(interx+20,20-interx)))
plt.xlabel('tiempo');plt.ylabel('Flujo de entrada')

plt.subplot(2,2,4);plt.grid(True)
plt.xlim((0,2*n_perts))
plt.title('Control derivativo')
plt.plot(tl,td,'b.')
plt.xlabel('tiempo');plt.ylabel('X')

plt.show()
        
        
