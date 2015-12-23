# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:13:00 2015

@author: carlosaranda
Pruebas de entrenamiento de una red neuronal con condiciones encontradas por un algoritmo 
genético para un modelo de un reboiler
"""
from __future__ import division, print_function

from columna2 import reboiler
from AGmultivar import AG
from RN import RN, redimensionarPesos, dimensionarMatricesPesos
import numpy as np
import time
import matplotlib.pyplot as plt 


def norm(a,maxmin=None):
    if not maxmin:
        minimo=np.min(a)[0]
        maximo=np.max(a)[0]
    else:
        maximo=maxmin[0];minimo=maxmin[1]
    if not isinstance(a,np.ndarray):
        a=np.array(a)
    return((a-minimo)/(maximo-minimo),(maximo,minimo))

def perturbacion(pert=None,Lvar0=None):
    Lvar=np.copy(Lvar0)
    #print('Perturbaciones de:')
    for i in range(1):
        inicio=0
        inter=tf-inicio
        fin=int(inicio+inter)
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

def f_obj(controladores):
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
AGS.parametros(optim=0,Nind=20,Ngen=100)
AGS.variables(comun=[4,0,4])

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
n_perturbaciones=2 #6max
n_pruebas=3
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
    tabla_resultados.append([Lvar,resultados_plot[0],resultados_plot[1]])
#    plt.plot(tabla_resultados[prueba][0])
#    plt.show()    
    t2=time.time()
    print('Tiempo por prueba:',(t2-t1)/60)


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
#    plt.plot(x)
#    plt.show()
    y=np.zeros(len(tabla_resultados))#Se crea una matriz de zeros del tamaño de la tabla de resultados,
                                     #así la red será entrenada para escojer a un conjunto de elementos
                                     #dependiendo del parecido de los datos de entrada con los de
                                     #entrenamiento
    y[cont]=1
    #y,maxmin_y=norm(res[1],(ymax,ymin))
    x_ent.append(x);x_maxmin.append(maxmin_x)
    y_ent.append(y)
    #y_ent.append(y);y_maxmin.append(maxmin_y)
    cont+=1

print(y_ent)
#print('lenY_ent:',len(y_ent))
#print('X_ent:',len(x_ent),len(x_ent[0]))
#print('Y_maxmin:',y_maxmin)  
#print('Xmaxmin:',x_maxmin)  
  
''' 
Entrenamiento de la red neuronal
'''
pImp=([3,3],50,500)
datos=[x_ent,y_ent]
print('\nInicio de la red')
red=RN()
ar=[pImp[0][0],pImp[0][1]]
red.parametros(arq=ar)

red.datosEnt(datos)

est,nPesos=dimensionarMatricesPesos(red.arq)
print('Arquitectura de red:',red.arq)
AGR=AG()
AGR.parametros(optim=0,Nind=pImp[1],Ngen=pImp[2])
def fobjR(pesos,est):
    W=redimensionarPesos(pesos,est)
    y,error=red.FP(W=W)
    return(error)

AGR.variables(comun=[nPesos,-30,30])
AGR.Fobj(fobjR,est)
pesos=list()
pesos=[[],1000]
print('\nComienzo de entrenamiento de la RN')

for prueba in range(n_pruebas):
    print('Prueba {}'.format(prueba+1))
    t1=time.time()
    Went,error=AGR.start()
    t2=time.time()
    if error<pesos[1]:
        pesos=[Went,error]
    print('Tiempo por prueba:',(t2-t1)/60)
    
print('\nFinalización de entrenamiento y almacenamiento de pesos')

with open('./pruebasent2.txt','a') as f:
    info='Datos de corrida:pImp['+str(pImp[0][0])+','+str(pImp[0][1])+','+str(pImp[1])+','+str(pImp[2])+str(dt)+']\n'
    f.write(info)
    p=''
    r=''
    for i in pesos[0]:
        p=p+str(i)+','
    for j in range(len(tabla_resultados)):
        r=r+'['
        for i in tabla_resultados[j][1]:
            r=r+str(i)+','
        r=r+'],'    
    w=p[:len(p)-1]+'/'+r[:len(r)-1]+'\n' #Todo junto
    f.write(w)
       
print('\nFin de la operación')



'''
  
f_obj(resultados_plot[0])
    
print('\nSe graficaran los siguientes resultados:',resultados_plot)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,17))
plt.title('Avance de simulacion')

plt.subplot(3,2,1)
plt.grid()
plt.plot(tl,Tl,'k.')
plt.xlabel('Tiempo');plt.ylabel('Temperatura')

plt.subplot(3,2,2)
plt.grid()
plt.plot(tl,Ml,'b-');plt.plot(tl,Bl,'g-');plt.plot(tl,Vl,'r.')
plt.xlabel('Tiempo');plt.ylabel('Masa')

plt.subplot(3,2,3)
plt.grid()
plt.plot(tl[1:],Lvar,'k.')
plt.xlabel('Tiempo');plt.ylabel('F')

plt.subplot(3,2,4)
plt.grid()
plt.plot(tl,xli,'g-')
plt.xlabel('Tiempo');plt.ylabel('X')
plt.show()
'''