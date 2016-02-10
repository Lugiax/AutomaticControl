# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:11:43 2016

@author: root
"""

from __future__ import division
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
###############################################################################
def Perturbaciones(rango,dt=0.01,n_perts=50,plot=0,tipo=None,seed=None,sin_pert=False):
    rnd.seed(seed);np.random.seed(seed)
    '''
    Generador aleatorio de funciones tipo perturbaciones en un rango (inicio,fin)
    '''
    def f_pert(x,mag,tipo,mitad):
        if tipo==1:#rampa
            return((x-inicio/l_L)*mag)
        elif tipo==2:
            return(mag)
        elif tipo==3:
            return (mag*np.exp(-(x-mitad)**2/(2*0.001**2)))
    
    x0,xmax=rango
    xmax/=dt
    L=(np.random.rand(xmax-x0)*2-1)*.1

    l_L=len(L)
    perts=list()
    for i in range(n_perts):
        L_p=np.copy(L)
        if sin_pert and rnd.random()>0.8:
            perts.append(L_p)
            if plot==1:
                plt.plot(L_p)
                plt.show()
            continue
        proporcion=rnd.random() #Proporcion de la perturbacion
        prop_inicio=rnd.random() #lugar del inicio de la perturbacion
        prop_magnitud=rnd.random() #proporcion de la magnitud
        if not tipo: tipo=rnd.randint(1,3) #Perturbación tipo:1-rampa, 2-escalon, 3-dirac
        
        if tipo==1: #Rampa
            magnitud=rnd.randint(-10,10)*prop_magnitud
        elif tipo==2:#Escalon
            magnitud=rnd.randint(-5,5)*prop_magnitud
        elif tipo==3:#Dirac
            magnitud=rnd.randint(-10,10)*prop_magnitud
    
        inicio=int((1-proporcion)*prop_inicio*l_L)
        num_datos_modificados=int(l_L*proporcion)
        
        for j in range(num_datos_modificados):
            L_p[inicio+j]+=f_pert((inicio+j)/l_L,magnitud,tipo,(num_datos_modificados/2+inicio)/l_L)
        
        L_p[inicio+j+1:]+=L_p[inicio+j]
        
        perts.append(L_p)
        if plot==1:
            plt.plot(L_p)
            plt.show()
    
    return(perts)
    
###############################################################################   
    
def Perturbar(pert=None,Lvar0=None,inter=(0,1),dt=0.01):
    '''
    Generador de perturbaciones en un arreglo Lvar0 en un intervalo (inicio,fin)
    '''
    Lvar=np.copy(Lvar0)
    gauss= lambda x:10*np.exp(-(x-0.5)**2/(2*0.05**2))
    for i in range(1):
        inicio=inter[0]
        intervalo=inter[1]-inter[0]
        fin=int(inicio+intervalo)
        x=0
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
    
###############################################################################

def norm(a,maxmin=None):
    arr=np.copy(a)
    if not maxmin:
        minimo=np.min(arr)
        maximo=np.max(arr)
        return((arr-minimo)/(maximo-minimo),(maximo,minimo))
    else:
        maximo=maxmin[0];minimo=maxmin[1]
        return((arr-minimo)/(maximo-minimo))

def denorm(a,maxmin):
    maximo,minimo=maxmin
    arr=np.copy(a)
    return(arr*(maximo-minimo)+minimo)

    