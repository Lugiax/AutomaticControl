# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:45:43 2015

@author: carlosaranda
Prueba final Red Neuronal para control
"""
import re
from columna2 import reboiler
from RN import RN, redimensionarPesos, dimensionarMatricesPesos
import numpy as np
import time
import matplotlib.pyplot as plt

###############################################################

def norm(a,maxmin=None):
    if not maxmin:
        minimo=np.min(a)
        maximo=np.max(a)
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
    
gauss= lambda x:10*np.exp(-(x-0.5)**2/(2*0.05**2))    
    
################################################################
with open('pruebasent.txt','r') as f:
    archivo=f.readlines()
    
datos=list()
cont=-1
for i in archivo:
    if re.search('^Datos',i):
        datos.append(list())
        pImp=re.findall('[0-9]+',i)
        pImp=[int(x) for x in pImp]
        cont+=1
    else:
        pesos=(re.findall('([-,0-9.]+)/',i))[0].split(',')
        pesos=[float(x) for x in pesos]
        resp1=re.findall('\[([-,0-9.]+),\]',i)
        resp=list()
        for i in resp1:
            i=i.split(',')
            i=[float(x) for x in i]
            resp.append(i)

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
reb.kcb=0;reb.tdb=0;reb.Bref=9 ##Para fondos
reb.kcq=0;reb.tdq=0 ##Para reboiler
reb.Mref=30. ##Para la masa del interior del reboiler
dt=0.01;tf=3;t=0
Ml,Bl,Vl,tl,Tl=None,None,None,None,None

## Condiciones iniciales:
reb.L=20;reb.xl=0.8
reb.M=30.;reb.Q=1.1e4;reb.B=9
## Estado Estacionario: reb.M=30;reb.Q=1.08e4;reb.B=9;reb.V=11
reb.x=reb.xl;reb.y=reb.equil(reb.x);lamvap=reb.lamvap_f(reb.x)
reb.hl=220.;reb.h=reb.hl;reb.H=reb.h+lamvap
reb.T=reb.h/reb.cp(reb.x);reb.V=reb.Q/reb.lamvap_f(reb.x)
reb.Pt=760.##mmHg
t=0
Ml,Bl,Vl,tl,Tl,xli=[reb.M],[reb.B],[reb.V],[t],[reb.T],[reb.x]

#Se inicia el arreglo del flujo de entrada
np.random.seed(10)
Lvar=np.random.random(int(tf/dt))+20

#se añaden las perturbaciones
#Lvar=perturbacion(0,Lvar,(0,1))
Lvar=perturbacion(0,Lvar,(1,2))
#Lvar=perturbacion(1,Lvar,(2,3))
#Lvar=perturbacion(5,Lvar,(3,4))
#Lvar=perturbacion(0,Lvar,(4,5))

maximo=np.max(Lvar)
minimo=np.min(Lvar)
print maximo,minimo
#plt.plot(Lvar0)
#plt.show()

'''
Se inicia la red neuronal
'''
red=RN()
ar=[pImp[0],pImp[1]]
red.parametros(arq=ar,arq_io=(int(1/dt),len(resp)))

est,nPesos=dimensionarMatricesPesos(red.arq)
W=redimensionarPesos(pesos,est)

for i in range(int(tf/dt)):
    if i < int(1/dt):
        continue
    else:
        ventana,minmax=norm([Lvar[i-1/dt:i]],(maximo,minimo))
        y,error=red.FP(W=W,xi=ventana)
        if np.sum(y)>=1:print np.around(y)
        if i%20==0:
            plt.title('{}-{}'.format(i*dt-1,i*dt))
            plt.plot(ventana[0])
            print np.around(y)[0]
            plt.show()
        
    t+=dt
    reb.L=float(Lvar[i])
    reb.actualizar(t,dt)
    Ml.append(reb.M),Bl.append(reb.B),Vl.append(reb.V),Tl.append(reb.T),tl.append(t),xli.append(reb.x)
        
